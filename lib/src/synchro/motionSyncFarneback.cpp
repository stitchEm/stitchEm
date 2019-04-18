// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "motionSyncFarneback.hpp"
#include "gpu/memcpy.hpp"
#include "image/unpack.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

namespace VideoStitch {
namespace Synchro {

bool cvFlow2MotionVectorField(const cv::Mat& flow, Motion::ImageSpace::MotionVectorField& field) {
  field.clear();
  if (flow.type() != CV_32FC2) {
    Logger::get(Logger::Error) << "cvFlow2MotionVectorField() : flow type must be CV_32FC2" << std::endl;
    return false;
  }

  field.reserve(flow.rows * flow.cols);

  for (int lin = 0; lin < flow.rows; ++lin) {
    for (int col = 0; col < flow.cols; ++col) {
      auto f = flow.at<cv::Vec2f>(lin, col);
      float2 from, to;
      from.x = static_cast<float>(col);
      from.y = static_cast<float>(lin);
      to.x = static_cast<float>(col) + f[0];
      to.y = static_cast<float>(lin) + f[1];
      field.push_back(Motion::ImageSpace::MotionVector(from, to));
    }
  }
  return true;
}

MotionEstimationTaskFarneback::MotionEstimationTaskFarneback(
    Util::Algorithm::ProgressReporter* progress, int64_t firstFrame, int64_t lastFrame, std::size_t nbInputs,
    const std::vector<std::shared_ptr<Input::VideoReader> >& readers,
    const std::vector<const Core::InputDefinition*>& inputDefs, const std::vector<std::size_t>& inputIndexesToProcess,
    std::vector<Motion::AffineMotionModelEstimation::MotionModel>& motionModels,
    std::vector<std::vector<double> >& magnitudes,
    std::vector<std::queue<VideoStitch::Motion::OpticalFlow> >& opticalFlowFields,
    std::vector<int>& numberOfProcessedFrames, std::atomic<int>& frameCounter, std::atomic<int>& cancellation,
    std::atomic<int>& failure, std::mutex& modelLock, int downScaleFactor, std::size_t maxFlowFieldSize,
    bool filterSmallMotions)
    : status(Status::OK()),
      progress(progress),
      firstFrame(firstFrame),
      lastFrame(lastFrame),
      nbInputs(nbInputs),
      readers(readers),
      inputDefs(inputDefs),
      inputIndexesToProcess(inputIndexesToProcess),
      motionModels(motionModels),
      magnitudes(magnitudes),
      opticalFlowFields(opticalFlowFields),
      numberOfProcessedFrames(numberOfProcessedFrames),
      frameCounter(frameCounter),
      cancellation(cancellation),
      failure(failure),
      height(0),
      width(0),
      modelLock(modelLock),
      downScaleFactor(downScaleFactor),
      maxFlowFieldSize(maxFlowFieldSize),
      filterSmallMotions(filterSmallMotions) {
  if (downScaleFactor < 1) {
    status = {Origin::MotionEstimationAlgorithm, ErrType::ImplementationError, "Unsupported downscale factor"};
    return;
  }

  if (nbInputs == 0) {
    status = {Origin::MotionEstimationAlgorithm, ErrType::InvalidConfiguration, "No inputs found"};
    return;
  }
  if (nbInputs != readers.size()) {
    status = {Origin::MotionEstimationAlgorithm, ErrType::ImplementationError,
              "Mismatching input size: " + std::to_string(nbInputs) + ", readers: " + std::to_string(readers.size())};
    return;
  }

  int64_t heightFirstInput = readers.front()->getHeight();
  int64_t widthFirstInput = readers.front()->getWidth();
  for (std::size_t i = 1; i < readers.size(); ++i) {
    if ((readers[i]->getHeight() != heightFirstInput) || (readers[i]->getWidth() != widthFirstInput)) {
      status = {Origin::MotionEstimationAlgorithm, ErrType::InvalidConfiguration, "All inputs must have the same size"};
      return;
    }
  }

  height = readers.front()->getHeight() / downScaleFactor;
  width = readers.front()->getWidth() / downScaleFactor;

  Logger::get(Logger::Verbose) << "Dimension of flow im: " << height << " x " << width << std::endl;

  auto potHostBuffer =
      GPU::HostBuffer<unsigned char>::allocate(readers.front()->getFrameDataSize(), "Motion estimation input frame");
  if (!potHostBuffer.ok()) {
    status = potHostBuffer.status();
    return;
  }
  hostBuffer = potHostBuffer.value();
  if (readers.front()->getSpec().addressSpace == Device) {
    auto potDevBuffer =
        GPU::Buffer<unsigned char>::allocate(readers.front()->getFrameDataSize(), "Motion estimation input frame");
    if (!potDevBuffer.ok()) {
      status = potDevBuffer.status();
      return;
    }
    devBuffer = potDevBuffer.value();
  }
  bufferResizedFirstFrame.resize(height * width);
  bufferResizedSecondFrame.resize(height * width);
  flow = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_32FC2);
}

MotionEstimationTaskFarneback::~MotionEstimationTaskFarneback() {
  if (devBuffer.wasAllocated()) {
    devBuffer.release();
  }
  hostBuffer.release();
}

void MotionEstimationTaskFarneback::run() {
  GPU::useDefaultBackendDevice();

  for (std::size_t curInd = 0; curInd < inputIndexesToProcess.size(); ++curInd) {
    std::size_t indexSource = inputIndexesToProcess[curInd];

    int64_t currentFrame = firstFrame;
    if (!loadFrame(*(readers[indexSource])).ok()) {
      Logger::get(Logger::Error) << "MotionEstimationTaskFarneback::run(): could not read frame " << currentFrame
                                 << " for source " << indexSource << std::endl;
      status = {Origin::MotionEstimationAlgorithm, ErrType::RuntimeError, "Could not load a frame"};
      failure++;
      return;
    }
    std::swap(bufferResizedFirstFrame,
              bufferResizedSecondFrame);  ///< the first frame goes into: bufferResizedFirstFrame
    currentFrame++;

    int indexCurrentFlow = 0;
    while (currentFrame <= lastFrame) {
      if (!loadFrame(*(readers[indexSource])).ok()) {
        Logger::get(Logger::Error) << "MotionEstimationTaskFarneback::run(): could not read frame " << currentFrame
                                   << " for source " << indexSource << std::endl;
        status = {Origin::MotionEstimationAlgorithm, ErrType::RuntimeError, "Could not load a frame"};
        failure++;
        return;
      }
      int flags = cv::OPTFLOW_USE_INITIAL_FLOW;
      if (currentFrame == firstFrame + 1) {
        flags = 0;  ///< first time we call the optical flow
      }

      currentFrame++;

      cv::Mat firstFrameCV(static_cast<int>(height), static_cast<int>(width), CV_8UC1, &(bufferResizedFirstFrame[0]));
      cv::Mat secondFrameCV(static_cast<int>(height), static_cast<int>(width), CV_8UC1, &(bufferResizedSecondFrame[0]));

      cv::calcOpticalFlowFarneback(firstFrameCV, secondFrameCV, flow, 0.5, 3, 15, 3, 5, 1.2, flags);
      std::swap(bufferResizedFirstFrame, bufferResizedSecondFrame);
      frameCounter++;
      numberOfProcessedFrames[indexSource]++;

      Motion::ImageSpace::MotionVectorField motionField;
      bool cv2MF = cvFlow2MotionVectorField(flow, motionField);
      if (!cv2MF) {
        status = {Origin::MotionEstimationAlgorithm, ErrType::ImplementationError,
                  "Could not convert the flow into a motion vector field"};
        return;
      }

      VideoStitch::Motion::OpticalFlow of(static_cast<int>(indexSource), (int)(firstFrame + indexCurrentFlow),
                                          motionField, inputDefs[indexSource]);
      indexCurrentFlow++;

      if (!magnitudes.empty()) {
        double medianMagnitude = sqrt(of.computeMedianMagnitude2());
        magnitudes.at(of.input).at(of.frame - firstFrame) = medianMagnitude;
      }

      Matrix33<double> h;
      Status statusMotionModel = Status::OK();
      if (!motionModels.empty()) {
        statusMotionModel = Motion::AffineMotionModelEstimation::motionModel(of.field, h, *of.inputDef);
      }

      if (failure > 0) {
        status = {Origin::MotionEstimationAlgorithm, ErrType::RuntimeError, "Could not load a frame"};
        return;
      }

      if (cancellation > 0) {
        status = {Origin::MotionEstimationAlgorithm, ErrType::OperationAbortedByUser,
                  "Motion sync algorithm cancelled"};
        return;
      }

      {
        std::unique_lock<std::mutex> sl(modelLock);

        if (!motionModels.empty()) {
          motionModels[of.input][of.frame - firstFrame] = std::make_pair(statusMotionModel.ok(), h);
        }

        if (!opticalFlowFields.empty()) {
          if (filterSmallMotions) {
            of.filterSmallMotions();
          }
          if (maxFlowFieldSize > 0) {
            of.sampleMotionVectors(maxFlowFieldSize);
          }
          of.applyFactor(static_cast<float>(downScaleFactor));
          opticalFlowFields.at(indexSource).push(of);
        }

        std::stringstream ss;
        ss << "Processing frame " << (frameCounter / nbInputs) << " out of " << lastFrame - firstFrame;
        if (cancellation > 0 ||
            (progress && progress->notify(ss.str(), (100.0 * (double)frameCounter) /
                                                        ((double)nbInputs * (double)(lastFrame - firstFrame))))) {
          ++cancellation;
          return;
        }
      }
    }
  }
}

Status MotionEstimationTaskFarneback::loadFrame(Input::VideoReader& reader) {
  unsigned char* bufferResizedPtr = &(bufferResizedSecondFrame[0]);

  unsigned char* origFrame = nullptr;
  switch (reader.getSpec().addressSpace) {
    case Device:
      origFrame = devBuffer.devicePtr();
      break;
    case Host:
      origFrame = hostBuffer.hostPtr();
      break;
  }
  mtime_t date;
  Input::ReadStatus statusRead = reader.readFrame(date, origFrame);
  if (!statusRead.ok()) {
    // TODOLATERSTATUS handle ReadStatus
    return {Origin::Input, ErrType::RuntimeError,
            "MotionEstimationTaskFarneback::loadFrame() : Could not read the frame"};
  }

  // transfer to host if needed
  switch (reader.getSpec().addressSpace) {
    case Device:
      FAIL_RETURN(GPU::memcpyBlocking(hostBuffer, devBuffer.as_const()));
      break;
    case Host:
      break;
  }

  // Colorspace conversion
  switch (reader.getSpec().format) {
    case VideoStitch::YUV422P10:
    case VideoStitch::YV12:
    case VideoStitch::NV12:
    case VideoStitch::Grayscale:
      break;
    default:
      status = {Origin::MotionEstimationAlgorithm, ErrType::ImplementationError,
                "Unimplemented for pixel format: " + std::string(getStringFromPixelFormat(reader.getSpec().format))};
      return status;
  }

  cv::Mat mat(static_cast<int>(reader.getHeight()), static_cast<int>(reader.getWidth()), CV_8UC1, hostBuffer.hostPtr());
  // Downsampling
  cv::Mat mat2(static_cast<int>(height), static_cast<int>(width), CV_8UC1, bufferResizedPtr);
  cv::resize(mat, mat2, cv::Size(mat2.cols, mat2.rows), 0, 0, cv::INTER_LINEAR);
  return Status::OK();
}

}  // end namespace Synchro
}  // end namespace VideoStitch
