// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rotationStabilization.hpp"

#include "motion/affineMotion.hpp"
#include "synchro/motionSyncFarneback.hpp"
#include "common/queue.hpp"
#include "common/thread.hpp"
#include "util/registeredAlgo.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/ptv.hpp"

#include <opencv2/core/core.hpp>

#include <atomic>
#include <sstream>
#ifndef NDEBUG
#include <iostream>
#endif

namespace VideoStitch {

namespace Stab {
namespace {
Util::RegisteredAlgo<RotationStabilizationAlgorithm> registered("stabilization");
}

RotationStabilizationAlgorithm::RotationStabilizationAlgorithm(const Ptv::Value* config)
    : firstFrame(0), lastFrame(1000), convolutionSpan(30) {
  if (config != NULL) {
    const Ptv::Value* value = config->has("first_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      firstFrame = value->asInt();
    }
    value = config->has("last_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      lastFrame = value->asInt();
    }
    value = config->has("devices");
    if (value && value->getType() == Ptv::Value::LIST) {
      const std::vector<Ptv::Value*>& devIds = value->asList();
      for (std::vector<Ptv::Value*>::const_iterator d = devIds.begin(); d != devIds.end(); ++d) {
        value = (*d)->has("id");
        if (value && value->getType() == Ptv::Value::INT) {
          devices.push_back((int)value->asInt());
        }
      }
    }
    value = config->has("convolution_span");
    if (value && value->getType() == Ptv::Value::INT) {
      convolutionSpan = value->asInt();
    }
    if (devices.size() == 0) devices.push_back(0);
  }
}

RotationStabilizationAlgorithm::~RotationStabilizationAlgorithm() {}

const char* RotationStabilizationAlgorithm::docString =
    "An algorithm that attemps to smooth camera movements with jitter.\n"
    "The configuration is as follow:\n"
    "{ \"first_frame\"       # The first frame of the sequence to stabilize\n"
    "  \"last_frame\"        # The last frame of the sequence to stabilize\n"
    "  \"convolution_span\"  # The radius in frame of the convolution\n"
    "                        # This corresponds to the low-pass frequency of the non-jitter movement,\n"
    "                        # eg. for removing vibrations at a frequency of 6 frames/sec, set the\n"
    "                        # radius to 6 or more.\n"
    "}\n";

namespace {
struct OpticalFlow {
  OpticalFlow() : frame(0) {}

  OpticalFlow(int64_t fr,
              std::vector<std::pair<Motion::ImageSpace::MotionVectorField, const Core::InputDefinition*> >& fi)
      : frame(fr), field(fi) {}

  int64_t frame;
  std::vector<std::pair<Motion::ImageSpace::MotionVectorField, const Core::InputDefinition*> > field;
};

struct OpticalFlowCompare {
  bool operator()(const OpticalFlow& lhs, const OpticalFlow& rhs) const { return lhs.frame > rhs.frame; }
};

class RotationModelFitting : public ThreadPool::Task {
 public:
  typedef Motion::RotationalMotionModelEstimation::MotionModel MotionModel;

  RotationModelFitting(const Core::PanoDefinition& pano, int64_t firstFrame, int64_t lastFrame,
                       Util::Algorithm::ProgressReporter* progress,
                       std::vector<std::queue<VideoStitch::Motion::OpticalFlow> >& opticalFlowFields,
                       std::mutex& queuesLock, MotionModel& model, std::mutex& modelLock, std::atomic<int>& frameCnt,
                       std::atomic<int>& cancellation)
      : estimator(pano),
        firstFrame(firstFrame),
        lastFrame(lastFrame),
        progress(progress),
        opticalFlowFields(opticalFlowFields),
        queuesLock(queuesLock),
        model(model),
        modelLock(modelLock),
        frameCounter(frameCnt),
        cancellation(cancellation) {}

  virtual void run() {
    for (;;) {
      if (cancellation) {
        return;
      }
      if (frameCounter == (lastFrame - firstFrame)) {
        return;
      }
      std::vector<std::pair<Motion::ImageSpace::MotionVectorField, const Core::InputDefinition*> > field;
      int currentFrame = 0;
      {
        std::unique_lock<std::mutex> ql(queuesLock);
        bool atLeastOneEmptyQueue = false;
        for (std::size_t i = 0; i < opticalFlowFields.size(); ++i) {
          atLeastOneEmptyQueue = atLeastOneEmptyQueue || opticalFlowFields[i].empty();
        }
        if (atLeastOneEmptyQueue) {
          continue;
        }

        currentFrame = opticalFlowFields.front().front().frame;
        bool allConsistent = true;
        for (std::size_t i = 1; i < opticalFlowFields.size(); ++i) {
          allConsistent = allConsistent && (currentFrame == opticalFlowFields[i].front().frame);
        }
        if (!allConsistent) {
          /// This should never happen
          std::ostringstream oss;
          oss << "RotationModelFitting: the set of processing queues is in an inconsistent state: ";
          for (std::size_t i = 0; i < opticalFlowFields.size(); ++i) {
            oss << opticalFlowFields[i].front().frame << "  ";
          }
          oss << "  Abording";
          Logger::get(Logger::Error) << oss.str() << std::endl;
          ++cancellation;
          return;
        }

        for (std::size_t i = 0; i < opticalFlowFields.size(); ++i) {
          field.push_back(std::make_pair(opticalFlowFields[i].front().field, opticalFlowFields[i].front().inputDef));
          opticalFlowFields[i].pop();
        }
      }

      // estimate the quaternion between last frame and current frame
      Quaternion<double> h;
      if (!estimator.motionModel(field, currentFrame, h).ok()) {
        Logger::get(Logger::Warning) << "Could not estimate rotation for frame " << currentFrame << ", skipping."
                                     << std::endl;
      }
      ++frameCounter;
      std::unique_lock<std::mutex> sl(modelLock);
      model[currentFrame] = h;
      std::stringstream ss;
      ss << "Computing rotational model for frame " << frameCounter << " out of " << lastFrame - firstFrame;
      if (cancellation > 0 ||
          (progress && progress->notify(ss.str(), (100.0 * (double)frameCounter) / (double)(lastFrame - firstFrame)))) {
        ++cancellation;
        return;
      }
    }
  }

 private:
  Motion::RotationalMotionModelEstimation estimator;
  int64_t firstFrame;
  int64_t lastFrame;
  Util::Algorithm::ProgressReporter* progress;
  std::vector<std::queue<VideoStitch::Motion::OpticalFlow> >& opticalFlowFields;
  std::mutex& queuesLock;
  MotionModel& model;
  std::mutex& modelLock;
  std::atomic<int>& frameCounter;
  std::atomic<int>& cancellation;
};
}  // namespace

namespace {
#ifndef NDEBUG
double radToDeg(double v) { return v * (180.0 / M_PI); }
#endif

inline cv::Vec4d quat2vec(Quaternion<double>& q) { return cv::Vec4d(q.getQ0(), q.getQ1(), q.getQ2(), q.getQ3()); }
}  // namespace

/**
 * For a sequence of unit quaternions qk, qk+1, qk+2, ...
 * with q = (cos(θk),sin(θk)nk)
 * Note that qk and − qk represent the same rotation (double folding property)
 * We need to first ensure that qk · ql > 0.
 * Now we can simply average them!
 *
 * Apply a temporal convolution, followed by a normalisation to unit length.
 * The length of the convolution can be a parameter.
 */
Potential<Ptv::Value> RotationStabilizationAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                            Util::OpaquePtr** ctx) const {
  std::mutex modelMutex;
  std::mutex queuesMutex;
  std::atomic<int> frameCounterOpticalFlow(0);
  std::atomic<int> frameCounterRotationEstimation(0);
  std::atomic<int> cancellation(0);
  std::atomic<int> failure(0);
  int numCores =
      getNumCores() < static_cast<int>(pano->numInputs()) ? getNumCores() : static_cast<int>(pano->numInputs());
  Logger::get(Logger::Warning) << "Num cores: " << numCores << std::endl;
  ThreadPool opticalFlowThreadPool(numCores);

  std::vector<int> numberOfProcessedFrames(pano->numInputs());
  Input::DefaultReaderFactory readerFactory((int)firstFrame, (int)lastFrame);
  std::vector<std::shared_ptr<Input::VideoReader> > readers;
  std::vector<const Core::InputDefinition*> inputDefs;

  frameid_t realLastFrame = static_cast<frameid_t>(lastFrame);
  std::vector<std::vector<std::size_t> > vectInputsPerCore(numCores);
  for (readerid_t indexSource = 0; indexSource < pano->numInputs(); ++indexSource) {
    vectInputsPerCore[indexSource % numCores].push_back(indexSource);

    const Core::InputDefinition* im = &(pano->getInput(indexSource));
    Potential<Input::Reader> reader = readerFactory.create(indexSource, *im);
    FAIL_CAUSE(reader.status(), Origin::StabilizationAlgorithm, ErrType::SetupFailure,
               "Could not create input readers");

    Input::VideoReader* videoReader = dynamic_cast<Input::VideoReader*>(reader.release());
    if (videoReader) {
      readers.push_back(std::shared_ptr<Input::VideoReader>(videoReader));
    }
    frameid_t lastFrameCurrentReader = videoReader->getLastFrame() - pano->getInput(indexSource).getFrameOffset();
    if (lastFrameCurrentReader < realLastFrame) {
      realLastFrame = lastFrameCurrentReader;
    }
    inputDefs.push_back(im);
  }
  if (realLastFrame < static_cast<frameid_t>(lastFrame)) {
    std::ostringstream oss;
    oss << "Last frame out of range. Updating last frame from " << lastFrame << " to " << realLastFrame << std::endl;
    Logger::get(Logger::Warning) << oss.str() << std::endl;
    lastFrame = realLastFrame;
  }

  std::vector<Motion::AffineMotionModelEstimation::MotionModel> motionModels;
  std::vector<std::vector<double> > magnitudes;
  std::vector<std::queue<VideoStitch::Motion::OpticalFlow> > opticalFlowFields(pano->numInputs());

  ThreadPool globalRotationEstimationThreadPool(numCores);

  int w = static_cast<int>(readers.front()->getWidth());
  int h = static_cast<int>(readers.front()->getHeight());
  int minDim = w < h ? w : h;

  int minSize = 128;
  int downscaleFactor = 1;
  while ((minDim / (2 * downscaleFactor)) > minSize) {
    downscaleFactor *= 2;
  }

  StabContext* state = NULL;
  if (ctx == NULL) {
    state = new StabContext();  // no memoization
  } else {
    if (*ctx == NULL) {
      *ctx = new StabContext();  // bootstrap memoization
    }
    state = dynamic_cast<StabContext*>(*ctx);
  }

  for (int indexCore = 0; indexCore < numCores; ++indexCore) {
    std::unique_ptr<Synchro::MotionEstimationTaskFarneback> taskFarneback(new Synchro::MotionEstimationTaskFarneback(
        progress, firstFrame, lastFrame, pano->numInputs(), readers, inputDefs, vectInputsPerCore[indexCore],
        motionModels, magnitudes, opticalFlowFields, numberOfProcessedFrames, frameCounterOpticalFlow, cancellation,
        failure, queuesMutex, downscaleFactor, 100, true));
    if (taskFarneback == nullptr) {
      return {Origin::StabilizationAlgorithm, ErrType::SetupFailure, "Could not initialize the motion estimation task"};
    }
    opticalFlowThreadPool.tryRun(taskFarneback.release());
  }

  for (int indexCore = 0; indexCore < numCores; ++indexCore) {
    std::unique_ptr<RotationModelFitting> taskRotationModel(
        new RotationModelFitting(*pano, firstFrame, lastFrame, nullptr, opticalFlowFields, queuesMutex, state->models,
                                 modelMutex, frameCounterRotationEstimation, cancellation));
    if (taskRotationModel == nullptr) {
      return {Origin::StabilizationAlgorithm, ErrType::SetupFailure, "Could not initialize the rotation model fitting"};
    }
    globalRotationEstimationThreadPool.tryRun(taskRotationModel.release());
  }

  opticalFlowThreadPool.waitAll();
  globalRotationEstimationThreadPool.waitAll();

#ifndef NDEBUG
  Logger::get(Logger::Debug) << "RotationStabilizationAlgorithm: frameCounterOpticalFlow: " << frameCounterOpticalFlow
                             << std::endl;
  Logger::get(Logger::Debug) << "RotationStabilizationAlgorithm: frameCounterRotationEstimation: "
                             << frameCounterRotationEstimation << std::endl;
  for (std::size_t i = 0; i < opticalFlowFields.size(); ++i) {
    Logger::get(Logger::Debug) << "RotationStabilizationAlgorithm: opticalFlowFields[" << i
                               << "] : " << opticalFlowFields[i].size() << std::endl;
  }
#endif

  if (cancellation) {
    return Status{Origin::StabilizationAlgorithm, ErrType::OperationAbortedByUser, "Algorithm cancelled"};
  }

  if (failure) {
    return Status{Origin::StabilizationAlgorithm, ErrType::OutOfResources, ""};
  }

  // From a motion model to a panorama-orientation model
  std::vector<Quaternion<double> > orientations;
  Quaternion<double> acc;
  const int64_t kUnknown = std::numeric_limits<size_t>::max();
  int64_t interpolate_from = kUnknown;
  for (int64_t i = firstFrame; i <= lastFrame; ++i) {
    const auto& m = state->models[i];
    if (m.getQ0() != 0.0 || m.getQ1() != 0.0 || m.getQ2() != 0.0 || m.getQ3() != 0.0) {
      if (interpolate_from != kUnknown) {
        // recover from interpolation mode
        // interpolate between last known and first correct quaternion
        for (int64_t j = 1; j < i - interpolate_from; ++j) {
          acc *= Quaternion<double>::slerp(state->models[(size_t)interpolate_from], m,
                                           (double)j / (double)(i - interpolate_from));
          orientations.push_back(acc.conjugate());
        }
        interpolate_from = kUnknown;
      }
      // standard case
      acc *= m;
      orientations.push_back(acc.conjugate());
    } else if (interpolate_from == kUnknown) {
      interpolate_from = i - 1;
    }
  }

  // Average the camera attitudes over the decay span,
  // use it as a target attitude
  // http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
  std::vector<Quaternion<double> > target;
  for (int i = 0; i < (int)orientations.size(); ++i) {
    cv::Matx<double, 4, 4> M = quat2vec(orientations[i]) * quat2vec(orientations[i]).t();
    for (int j = 1; j <= (int)convolutionSpan; j++) {
      if (i - j >= 0) M += quat2vec(orientations[i - j]) * quat2vec(orientations[i - j]).t();
      if (i + j < (int)orientations.size()) M += quat2vec(orientations[i + j]) * quat2vec(orientations[i + j]).t();
    }

#ifndef NDEBUG
    Logger::get(Logger::Debug) << "RotationStabilizationAlgorithm: Matrix M" << std::endl;
    for (int indexRow = 0; indexRow < M.rows; ++indexRow) {
      for (int indexCol = 0; indexCol < M.cols; ++indexCol) {
        Logger::get(Logger::Debug) << M(indexRow, indexCol) << "\t";
      }
      Logger::get(Logger::Debug) << std::endl;
    }
#endif

    // take the eigenvector of M with the biggest eigenvalue and normalize it
    cv::Matx<double, 4, 4> eigenvectors;
    cv::Matx<double, 4, 1> eigenvalues;
    cv::eigen(M, eigenvalues, eigenvectors);

#ifndef NDEBUG
    Logger::get(Logger::Debug) << "RotationStabilizationAlgorithm: eigenvectors:" << std::endl;
    for (int indexRow = 0; indexRow < eigenvectors.rows; ++indexRow) {
      for (int indexCol = 0; indexCol < eigenvectors.cols; ++indexCol) {
        Logger::get(Logger::Debug) << eigenvectors(indexRow, indexCol) << "\t";
      }
      Logger::get(Logger::Debug) << std::endl;
    }
    Logger::get(Logger::Debug) << std::endl
                               << "RotationStabilizationAlgorithm: eigenvalues: " << eigenvalues(0, 0) << " "
                               << eigenvalues(1, 0) << " " << eigenvalues(2, 0) << " " << eigenvalues(3, 0)
                               << std::endl;
#endif

    // eigen returns eigenvalues in descending order
    Quaternion<double> attitude(eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2), eigenvectors(0, 3));
    target.push_back(attitude / attitude.norm());
  }

#ifndef NDEBUG
  // GNUPlot command:
  // plot "plot.data" using 2 title "Yaw", "plot.data" using 3 title "Pitch", "plot.data" using 4 title "Roll"
  // plot "plot.data" using 2 title "Yaw", "plot.data" using 3 title "Pitch", "plot.data" using 4 title "Roll",
  // "plot.data" using 5 title "Corrected Yaw", "plot.data" using 6 title "Corrected Pitch", "plot.data" using 7 title
  // "Corrected Roll" plot "plot.data" using 2 title "Yaw", "plot.data" using 3 title "Pitch", "plot.data" using 4 title
  // "Roll", "plot.data" using 5 title "Target Yaw", "plot.data" using 6 title "Target Pitch", "plot.data" using 7 title
  // "Target Roll", "plot.data" using 8 title "Yaw Correction", "plot.data" using 9 title "Pitch Correction",
  // "plot.data" using 10 title "Roll Correction"
  for (size_t i = 0; i < orientations.size(); ++i) {
    double yaw, pitch, roll;
    double tar_yaw, tar_pitch, tar_roll;
    double corr_yaw, corr_pitch, corr_roll;
    Quaternion<double> r = orientations[i].conjugate() * target[i];
    orientations[i].toEuler(yaw, pitch, roll);
    target[i].toEuler(tar_yaw, tar_pitch, tar_roll);
    r.toEuler(corr_yaw, corr_pitch, corr_roll);
    Quaternion<double> fake = orientations[i].conjugate();
    double fcorr_yaw, fcorr_pitch, fcorr_roll;
    fake.toEuler(fcorr_yaw, fcorr_pitch, fcorr_roll);
    // std::cout << "Frame" << i << " " << radToDeg(yaw) << " " << radToDeg(pitch) << " " << radToDeg(roll);
    // std::cout << " " << radToDeg(tar_yaw) << " " << radToDeg(tar_pitch) << " " << radToDeg(tar_roll);
    // std::cout << " " << radToDeg(corr_yaw) << " " << radToDeg(corr_pitch) << " " << radToDeg(corr_roll) << std::endl;
    std::cout << radToDeg(roll) << " " << radToDeg(tar_roll) << " " << radToDeg(corr_roll) << " "
              << radToDeg(fcorr_roll) << std::endl;
  }
#endif

  // apply the rotation from the current orientation to the target orientation
  // we want the initial frame to have the same orientation as the stabilized
  // previous frame before correction
  // initial = orientations[firstFrame+1].conjugate() * target[firstFrame+1] * correction
  // correction = target[firstFrame+1].conjugate() * orientations[firstFrame+1] * initial
  Quaternion<double> initial = pano->getStabilization().at((int)firstFrame);
  Quaternion<double> correction = target[0].conjugate() * orientations[0] * initial;
  Core::SphericalSpline* head = Core::SphericalSpline::point((int)firstFrame, initial);
  Core::SphericalSpline* spline = head;
  for (size_t i = 0; i < orientations.size(); ++i) {
    Quaternion<double> r = orientations[i].conjugate() * target[i] * correction;
    spline = spline->lineTo((int)(firstFrame + i + 1), r);
  }
  Core::QuaternionCurve* stab = new Core::QuaternionCurve(head);

  stab->extend(&pano->getStabilization());
  pano->replaceStabilization(stab);
  Core::Curve *yaw = NULL, *pitch = NULL, *roll = NULL;
  Core::toEuler(pano->getStabilization(), &yaw, &pitch, &roll);
  pano->replaceStabilizationYaw(yaw);
  pano->replaceStabilizationPitch(pitch);
  pano->replaceStabilizationRoll(roll);

  if (ctx == NULL) {
    delete state;
  }

  return Potential<Ptv::Value>(Status::OK());
}

}  // namespace Stab
}  // namespace VideoStitch
