// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputMaskInterpolation.hpp"

#include "common/container.hpp"
#include "core1/bounds.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/memcpy.hpp"
#include "util/pngutil.hpp"
#include "util/geometryProcessingUtils.hpp"
#include "mask/mergerMask.hpp"
#include "libvideostitch/input.hpp"

#include <queue>
#include <stack>

//#define INPUTMASKINTERPOLATION_DEBUG

#if defined(INPUTMASKINTERPOLATION_DEBUG)
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/pngutil.hpp"
#include "../util/pnm.hpp"
#include "../util/debugUtils.hpp"
#include "../util/drawingUtils.hpp"
#endif

namespace VideoStitch {
namespace MaskInterpolation {

Potential<InputMaskInterpolation> InputMaskInterpolation::create(
    const Core::PanoDefinition& pano, const std::map<readerid_t, Input::VideoReader*>& readers,
    const int polygonSampleCount) {
  std::unique_ptr<InputMaskInterpolation> inputMaskInterpolation;
  inputMaskInterpolation.reset(new InputMaskInterpolation(polygonSampleCount));
  FAIL_RETURN(inputMaskInterpolation->setup(pano, readers));
  return inputMaskInterpolation.release();
}

InputMaskInterpolation::InputMaskInterpolation(const int polygonSampleCount)
    : frameId0(std::numeric_limits<frameid_t>::max()),
      frameId1(std::numeric_limits<frameid_t>::min()),
      polygonSampleCount(polygonSampleCount),
      activated(false) {}

InputMaskInterpolation::~InputMaskInterpolation() {
  deleteAllValues(transforms);
  delete devCoord;
}

void InputMaskInterpolation::deactivate() { activated = false; }

bool InputMaskInterpolation::isActive() { return activated; }

Status InputMaskInterpolation::setup(const Core::PanoDefinition& pano,
                                     const std::map<readerid_t, Input::VideoReader*>& readers) {
  int2 inputSize = make_int2(0, 0);
  const int inputScaleFactor(pano.getBlendingMaskInputScaleFactor());
  for (auto reader : readers) {
    const Core::InputDefinition& inputDef = pano.getInput(reader.second->id);
    Core::Transform* transform = Core::Transform::create(inputDef);
    if (!transform) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    transforms[reader.second->id] = transform;
    if (inputDef.getWidth() > inputSize.x) {
      inputSize.x = (int)inputDef.getWidth();
    }
    if (inputDef.getHeight() > inputSize.y) {
      inputSize.y = (int)inputDef.getHeight();
    }
  }
  auto tex =
      Core::OffscreenAllocator::createCoordSurface(pano.getWidth(), pano.getHeight(), "Input Mask Interpolation");
  if (!tex.ok()) {
    return tex.status();
  }
  devCoord = tex.release();

  FAIL_RETURN(
      inputMask.alloc(inputSize.x * inputSize.y * inputScaleFactor * inputScaleFactor, "Input mask interpolation"));
  masks.resize(inputSize.x * inputSize.y * inputScaleFactor * inputScaleFactor);
  return Status::OK();
}

std::pair<frameid_t, frameid_t> InputMaskInterpolation::getFrameIds() const { return {frameId0, frameId1}; }

Status InputMaskInterpolation::setupKeyframes(
#ifdef INPUTMASKINTERPOLATION_DEBUG
    const Core::PanoDefinition& pano,
#else
    const Core::PanoDefinition&,
#endif
    const frameid_t frameId0, const std::map<videoreaderid_t, std::vector<cv::Point>>& polygon0s,
    const frameid_t frameId1, const std::map<videoreaderid_t, std::vector<cv::Point>>& polygon1s) {

  if (frameId0 > frameId1) {
    return {Origin::MaskInterpolationAlgorithm, ErrType::InvalidConfiguration,
            "The second frame id is larger than the first"};
  }
  if (this->frameId0 == frameId0 && this->frameId1 == frameId1) {
    return Status::OK();
  }
  this->frameId0 = frameId0;
  this->frameId1 = frameId1;
  if (polygon0s.size() != polygon1s.size()) {
    return {Origin::MaskInterpolationAlgorithm, ErrType::InvalidConfiguration, "Number of polygons does not match"};
  }
  for (auto polygon0 : polygon0s) {
    auto polygon1 = polygon1s.find(polygon0.first);
    if (polygon1 == polygon1s.end()) {
      return {Origin::MaskInterpolationAlgorithm, ErrType::InvalidConfiguration, "Input polygon id do not match"};
    }

    std::vector<cv::Point2f> sampledPoint0 =
        Util::GeometryProcessing::getUniformSampleOnPolygon(polygon0.second, polygonSampleCount);
    std::vector<cv::Point2f> sampledPoint1 =
        Util::GeometryProcessing::getUniformSampleOnPolygon(polygon1->second, polygonSampleCount);
    sampledPoint0s.insert({(int)polygon0.first, sampledPoint0});
    sampledPoint1s.insert({(int)polygon0.first, sampledPoint1});
    std::vector<int> matchIndex;
    Util::GeometryProcessing::contourMatching(sampledPoint0, sampledPoint1, matchIndex);
    matchIndices.insert({polygon0.first, matchIndex});

#ifdef INPUTMASKINTERPOLATION_DEBUG
    {
      const int imId = polygon0.first;
      const Core::InputDefinition& inputDef = pano.getInput(imId);
      const int inputScaleFactor(pano.getBlendingMaskInputScaleFactor());
      const int2 inputSize = make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight());
      const int2 imageSize = make_int2(inputSize.x * inputScaleFactor, inputSize.y * inputScaleFactor);
      std::vector<unsigned char> masks(imageSize.x * imageSize.y);
      {
        std::stringstream ss;
        ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
              "maskinterpolation/";
        ss << "original polygon-" << imId << "-0.png";
        FAIL_RETURN(Util::GeometryProcessing::drawPolygon(cv::Size(imageSize.x, imageSize.y), polygon0.second, masks));
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), masks, imageSize.x, imageSize.y);
      }
      {
        std::stringstream ss;
        ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
              "maskinterpolation/";
        ss << "original polygon-" << imId << "-1.png";
        FAIL_RETURN(Util::GeometryProcessing::drawPolygon(cv::Size(imageSize.x, imageSize.y), polygon1->second, masks));
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), masks, imageSize.x, imageSize.y);
      }
      {
        std::stringstream ss;
        ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
              "maskinterpolation/";
        ss << "sampled point-" << imId << "-0.png";
        Util::GeometryProcessing::dumpPoints(ss.str(), imageSize.x, imageSize.y, sampledPoint0);
      }
      {
        std::stringstream ss;
        ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
              "maskinterpolation/";
        ss << "sampled point-" << imId << "-1.png";
        Util::GeometryProcessing::dumpPoints(ss.str(), imageSize.x, imageSize.y, sampledPoint1);
      }
      {
        std::stringstream ss;
        ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
              "maskinterpolation/";
        ss << "contour matching-" << imId << ".png";
        Util::GeometryProcessing::dumpContourMatch(ss.str(), imageSize.x, imageSize.y, sampledPoint0, sampledPoint1,
                                                   matchIndex);
      }
    }
#endif
  }
  activated = true;
  return Status::OK();
}

Status InputMaskInterpolation::getInputsMap(const Core::PanoDefinition& pano, const frameid_t frame,
                                            GPU::Buffer<uint32_t> inputsMap) const {
  std::map<videoreaderid_t, std::vector<cv::Point>> inputs;
  FAIL_RETURN(getInputs(frame, inputs));

#ifdef INPUTMASKINTERPOLATION_DEBUG
  {
    for (auto input : inputs) {
      const int imId = input.first;
      const Core::InputDefinition& inputDef = pano.getInput(imId);
      const int inputScaleFactor(pano.getBlendingMaskInputScaleFactor());
      const int2 inputSize = make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight());
      const int2 imageSize = make_int2(inputSize.x * inputScaleFactor, inputSize.y * inputScaleFactor);
      std::vector<unsigned char> masks(imageSize.x * imageSize.y);
      std::stringstream ss;
      ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/"
            "maskinterpolation/";
      ss << "interpolated polygon-" << imId << ".png";
      Util::GeometryProcessing::drawPolygon(cv::Size(imageSize.x, imageSize.y), input.second, masks);
      Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), masks, imageSize.x, imageSize.y);
    }
  }
#endif

  const int inputScaleFactor(pano.getBlendingMaskInputScaleFactor());
  Core::Rect outputBounds =
      Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, pano.getHeight() - 1, pano.getWidth() - 1);
  GPU::Stream stream = GPU::Stream::getDefault();
  FAIL_RETURN(GPU::memsetToZeroAsync(inputsMap, stream));
  for (auto transform : transforms) {
    const videoreaderid_t imId = transform.first;
    const Core::InputDefinition& inputDef = pano.getVideoInput(imId);
    const int2 inputSize = make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight());
    FAIL_RETURN(transform.second->mapBufferCoord(0, *devCoord->pimpl->surface, outputBounds, pano, inputDef, stream));
    std::vector<std::vector<cv::Point>> points;
    points.push_back(inputs[imId]);
    FAIL_RETURN(Util::GeometryProcessing::drawPolygon(
        cv::Size(inputSize.x * inputScaleFactor, inputSize.y * inputScaleFactor), points, masks));
    FAIL_RETURN(GPU::memcpyBlocking<unsigned char>(inputMask.borrow(), &masks[0], masks.size()));
    FAIL_RETURN(MergerMask::MergerMask::getOutputIndicesFromInputMask(
        imId, inputScaleFactor, inputSize, inputMask.borrow_const(),
        make_int2((int)outputBounds.getWidth(), (int)outputBounds.getHeight()), *devCoord->pimpl->surface, inputsMap,
        stream));
  }
  return Status::OK();
}

Status InputMaskInterpolation::getInputs(const frameid_t frame,
                                         std::map<videoreaderid_t, std::vector<cv::Point>>& inputs) const {
  inputs.clear();
  if (frame <= frameId0) {
    for (auto sampledPoint : sampledPoint0s) {
      std::vector<cv::Point> points;
      for (size_t i = 0; i < sampledPoint.second.size(); i++) {
        points.push_back(cv::Point(sampledPoint.second[i]));
      }
      inputs.insert({sampledPoint.first, points});
    }
  } else if (frame >= frameId1) {
    for (auto sampledPoint : sampledPoint1s) {
      std::vector<cv::Point> points;
      for (size_t i = 0; i < sampledPoint.second.size(); i++) {
        points.push_back(cv::Point(sampledPoint.second[i]));
      }
      inputs.insert({sampledPoint.first, points});
    }
  } else {
    float t = float(frame - frameId0) / (frameId1 - frameId0);
    for (auto sampledPoint0 : sampledPoint0s) {
      std::vector<cv::Point> points;
      const videoreaderid_t imId = sampledPoint0.first;
      auto sampledPoint1 = sampledPoint1s.find(imId);
      auto matchIter = matchIndices.find(imId);
      if (matchIter == matchIndices.end()) {
        return {Origin::MaskInterpolationAlgorithm, ErrType::InvalidConfiguration, ""};
      }
      const auto& indices = matchIter->second;
      for (size_t i = 0; i < sampledPoint0.second.size(); i++) {
        points.push_back(cv::Point(sampledPoint0.second[i] * (1.0f - t) + sampledPoint1->second[indices[i]] * t));
      }
      inputs.insert({sampledPoint0.first, points});
    }
  }
  return Status::OK();
}

}  // namespace MaskInterpolation
}  // namespace VideoStitch
