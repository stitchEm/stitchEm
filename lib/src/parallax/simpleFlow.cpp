// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./simpleFlow.hpp"

#include "./mergerPair.hpp"
#include "./flowConstant.hpp"

#include "backend/common/imageOps.hpp"

#include "parse/json.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "util/opticalFlowUtils.hpp"
#include "util/imageProcessingGPUUtils.hpp"

#include "libvideostitch/parse.hpp"

//#define FLOW_DEBUG_IMAGE

#ifdef FLOW_DEBUG_IMAGE
#include "util/debugUtils.hpp"
#endif

#ifndef NDEBUG
#include "../util/debugUtils.hpp"
#include <sstream>
#endif

#define SIMPLE_FLOW_TILE_WIDTH 16
#define SIMPLE_FLOW_DEFAULT_FLOW_SIZE 15
#define SIMPLE_FLOW_DEFAULT_WINDOW_SIZE 3
#define SIMPLE_FLOW_DEFAULT_FLOW_GRADIENT_WEIGHT 0.4
#define SIMPLE_FLOW_DEFAULT_FLOW_MAGNITUDE_WEIGHT 0.2
#define SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_THRESH 0.15
#define SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_GAMMA 0.8
#define SIMPLE_FLOW_DEFAULT_FLOW_CONF_CLAMPED_VALUE 1
#define SIMPLE_FLOW_DEFAULT_FLOW_INT_KERNEL_SIZE 40
#define SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_SPACE 5
#define SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_IMAGE 5
#define SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_CONF 20
#define SIMPLE_FLOW_DEFAULT_FLOW_EXTRAPOLATION_KERNEL_SIZE 40
#define SIMPLE_FLOW_DEFAULT_UPSAMPLING_JITTER_SIZE 1
#define SIMPLE_FLOW_DEFAULT_LEFT_OFFSET 1
#define SIMPLE_FLOW_DEFAULT_RIGHT_OFFSET 0
#define SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_TIME 5

namespace VideoStitch {
namespace Core {
SimpleFlow::Factory::Factory(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                             const float gradientWeight, const float confidenceTransformThreshold,
                             const float confidenceTransformGamma, const float confidenceTransformClampedValue,
                             const int interpolationKernelSize, const float interpolationSigmaSpace,
                             const float interpolationSigmaImage, const float interpolationSigmaConfidence,
                             const int extrapolationKernelSize, const int upsamplingJitterSize, const int leftOffset,
                             const int rightOffset, const float interpolationSigmaTime)
    : flowSize(flowSize),
      windowSize(windowSize),
      flowMagnitudeWeight(flowMagnitudeWeight),
      gradientWeight(gradientWeight),
      confidenceTransformThreshold(confidenceTransformThreshold),
      confidenceTransformGamma(confidenceTransformGamma),
      confidenceTransformClampedValue(confidenceTransformClampedValue),
      interpolationKernelSize(interpolationKernelSize),
      interpolationSigmaSpace(interpolationSigmaSpace),
      interpolationSigmaImage(interpolationSigmaImage),
      interpolationSigmaConfidence(interpolationSigmaConfidence),
      extrapolationKernelSize(extrapolationKernelSize),
      upsamplingJitterSize(upsamplingJitterSize),
      leftOffset(leftOffset),
      rightOffset(rightOffset),
      interpolationSigmaTime(interpolationSigmaTime) {}

Potential<ImageFlowFactory> SimpleFlow::Factory::parse(const Ptv::Value& value) {
  int flowSize = SIMPLE_FLOW_DEFAULT_FLOW_SIZE;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "flowSize", flowSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration, "Invalid type for 'flowSize' configuration, expected int"};
  }
  int windowSize = SIMPLE_FLOW_DEFAULT_WINDOW_SIZE;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "windowSize", windowSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'windowSize' configuration, expected int"};
  }
  double flowMagnitudeWeight = SIMPLE_FLOW_DEFAULT_FLOW_MAGNITUDE_WEIGHT;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "flowMagnitudeWeight", flowMagnitudeWeight, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'flowMagnitudeWeight' configuration, expected double"};
  }
  double gradientWeight = SIMPLE_FLOW_DEFAULT_FLOW_GRADIENT_WEIGHT;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "gradientWeight", gradientWeight, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'gradientWeight' configuration, expected double"};
  }

  double confidenceTransformThreshold = SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_THRESH;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "confidenceTransformThreshold",
                            confidenceTransformThreshold, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'confidenceTransformThreshold' configuration, expected double"};
  }

  double confidenceTransformGamma = SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_GAMMA;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "confidenceTransformGamma", confidenceTransformGamma,
                            false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'confidenceTransformGamma' configuration, expected double"};
  }

  double confidenceTransformClampedValue = SIMPLE_FLOW_DEFAULT_FLOW_CONF_CLAMPED_VALUE;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "confidenceTransformClampedValue",
                            confidenceTransformClampedValue, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'confidenceTransformClampedValue' configuration, expected double"};
  }

  int interpolationKernelSize = SIMPLE_FLOW_DEFAULT_FLOW_INT_KERNEL_SIZE;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "interpolationKernelSize", interpolationKernelSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'interpolationKernelSize' configuration, expected int"};
  }

  double interpolationSigmaSpace = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_SPACE;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "interpolationSigmaSpace", interpolationSigmaSpace,
                            false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'interpolationSigmaSpace' configuration, expected double"};
  }

  double interpolationSigmaImage = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_IMAGE;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "interpolationSigmaImage", interpolationSigmaImage,
                            false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'interpolationSigmaImage' configuration, expected double"};
  }

  double interpolationSigmaConfidence = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_CONF;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "interpolationSigmaConfidence",
                            interpolationSigmaConfidence, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'interpolationSigmaConfidence' configuration, expected double"};
  }

  int extrapolationKernelSize = SIMPLE_FLOW_DEFAULT_FLOW_EXTRAPOLATION_KERNEL_SIZE;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "extrapolationKernelSize", extrapolationKernelSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'extrapolationKernelSize' configuration, expected int"};
  }

  int upsamplingJitterSize = SIMPLE_FLOW_DEFAULT_UPSAMPLING_JITTER_SIZE;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "upsamplingJitterSize", upsamplingJitterSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'upsamplingJitterSize' configuration, expected int"};
  }

  int leftOffset = SIMPLE_FLOW_DEFAULT_LEFT_OFFSET;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "leftOffset", leftOffset, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'leftOffset' configuration, expected int"};
  }

  int rightOffset = SIMPLE_FLOW_DEFAULT_RIGHT_OFFSET;
  if (Parse::populateInt("SimpleImageFlowFactory", value, "rightOffset", rightOffset, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'rightOffset' configuration, expected int"};
  }

  double interpolationSigmaTime = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_TIME;
  if (Parse::populateDouble("SimpleImageFlowFactory", value, "interpolationSigmaTime", interpolationSigmaTime, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'interpolationSigmaTime' configuration, expected double"};
  }

  return Potential<ImageFlowFactory>(new SimpleFlow::Factory(
      flowSize, windowSize, (float)flowMagnitudeWeight, (float)gradientWeight, (float)confidenceTransformThreshold,
      (float)confidenceTransformGamma, (float)confidenceTransformClampedValue, interpolationKernelSize,
      (float)interpolationSigmaSpace, (float)interpolationSigmaImage, (float)interpolationSigmaConfidence,
      extrapolationKernelSize, upsamplingJitterSize, leftOffset, rightOffset, (float)interpolationSigmaTime));
}

Ptv::Value* SimpleFlow::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue(SimpleFlow::getName()));
  return res;
}

bool SimpleFlow::Factory::needsInputPreProcessing() const { return true; }

std::string SimpleFlow::Factory::getImageFlowName() const { return SimpleFlow::getName(); }

std::string SimpleFlow::Factory::hash() const {
  std::stringstream ss;
  ss << "Simple Flow " << flowSize << " " << windowSize << " " << flowMagnitudeWeight << " " << gradientWeight << " "
     << confidenceTransformThreshold << " " << confidenceTransformGamma << " " << confidenceTransformClampedValue << " "
     << interpolationKernelSize << " " << interpolationSigmaSpace << " " << interpolationSigmaImage << " "
     << interpolationSigmaConfidence << extrapolationKernelSize << " " << leftOffset << " " << rightOffset
     << interpolationSigmaTime;
  return ss.str();
}

Potential<ImageFlow> SimpleFlow::Factory::create() const {
  std::map<std::string, float> parameters;
  parameters["flowSize"] = (float)flowSize;
  parameters["windowSize"] = (float)windowSize;
  parameters["flowMagnitudeWeight"] = flowMagnitudeWeight;
  parameters["gradientWeight"] = gradientWeight;

  parameters["confidenceTransformThreshold"] = confidenceTransformThreshold;
  parameters["confidenceTransformGamma"] = confidenceTransformGamma;
  parameters["confidenceTransformClampedValue"] = confidenceTransformClampedValue;

  parameters["interpolationKernelSize"] = (float)interpolationKernelSize;
  parameters["interpolationSigmaSpace"] = interpolationSigmaSpace;
  parameters["interpolationSigmaImage"] = interpolationSigmaImage;
  parameters["interpolationSigmaConfidence"] = interpolationSigmaConfidence;

  parameters["extrapolationKernelSize"] = (float)extrapolationKernelSize;

  parameters["upsamplingJitterSize"] = (float)upsamplingJitterSize;

  parameters["leftOffset"] = (float)leftOffset;
  parameters["rightOffset"] = (float)rightOffset;
  parameters["interpolationSigmaTime"] = (float)interpolationSigmaTime;
  return Potential<ImageFlow>(new SimpleFlow(parameters));
}

ImageFlowFactory* SimpleFlow::Factory::clone() const {
  return new Factory(flowSize, windowSize, flowMagnitudeWeight, gradientWeight, confidenceTransformThreshold,
                     confidenceTransformGamma, confidenceTransformClampedValue, interpolationKernelSize,
                     interpolationSigmaSpace, interpolationSigmaImage, interpolationSigmaConfidence,
                     extrapolationKernelSize, upsamplingJitterSize, leftOffset, rightOffset, interpolationSigmaTime);
}

SimpleFlow::SimpleFlow(const std::map<std::string, float>& parameters_) : ImageFlow(parameters_) {
  if (this->parameters.find("leftOffset") != this->parameters.end() &&
      this->parameters.find("rightOffset") != this->parameters.end()) {
    const int leftOffset = (int)this->parameters.find("leftOffset")->second;
    const int rightOffset = (int)this->parameters.find("rightOffset")->second;
    if (rightOffset >= leftOffset) {
      flowSequence.reset(new FlowSequence(leftOffset, rightOffset));
    }
  }
  if (parameters.find("flowSize") == parameters.end()) {
    parameters["flowSize"] = SIMPLE_FLOW_DEFAULT_FLOW_SIZE;
  }
  if (parameters.find("windowSize") == parameters.end()) {
    parameters["windowSize"] = SIMPLE_FLOW_DEFAULT_WINDOW_SIZE;
  }
  if (parameters.find("flowMagnitudeWeight") == parameters.end()) {
    parameters["flowMagnitudeWeight"] = (float)SIMPLE_FLOW_DEFAULT_FLOW_MAGNITUDE_WEIGHT;
  }
  if (parameters.find("gradientWeight") == parameters.end()) {
    parameters["gradientWeight"] = (float)SIMPLE_FLOW_DEFAULT_FLOW_GRADIENT_WEIGHT;
  }
  if (parameters.find("confidenceTransformThreshold") == parameters.end()) {
    parameters["confidenceTransformThreshold"] = (float)SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_THRESH;
  }
  if (parameters.find("confidenceTransformGamma") == parameters.end()) {
    parameters["confidenceTransformGamma"] = (float)SIMPLE_FLOW_DEFAULT_FLOW_CONF_TRANS_GAMMA;
  }
  if (parameters.find("confidenceTransformClampedValue") == parameters.end()) {
    parameters["confidenceTransformClampedValue"] = SIMPLE_FLOW_DEFAULT_FLOW_CONF_CLAMPED_VALUE;
  }
  if (parameters.find("interpolationKernelSize") == parameters.end()) {
    parameters["interpolationKernelSize"] = SIMPLE_FLOW_DEFAULT_FLOW_INT_KERNEL_SIZE;
  }
  if (parameters.find("interpolationSigmaSpace") == parameters.end()) {
    parameters["interpolationSigmaSpace"] = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_SPACE;
  }
  if (parameters.find("interpolationSigmaImage") == parameters.end()) {
    parameters["interpolationSigmaImage"] = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_IMAGE;
  }
  if (parameters.find("interpolationSigmaConfidence") == parameters.end()) {
    parameters["interpolationSigmaConfidence"] = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_CONF;
  }
  if (parameters.find("extrapolationKernelSize") == parameters.end()) {
    parameters["extrapolationKernelSize"] = SIMPLE_FLOW_DEFAULT_FLOW_EXTRAPOLATION_KERNEL_SIZE;
  }
  if (parameters.find("upsamplingJitterSize") == parameters.end()) {
    parameters["upsamplingJitterSize"] = SIMPLE_FLOW_DEFAULT_UPSAMPLING_JITTER_SIZE;
  }
  if (parameters.find("interpolationSigmaTime") == parameters.end()) {
    parameters["interpolationSigmaTime"] = SIMPLE_FLOW_DEFAULT_FLOW_INT_SIGMA_TIME;
  }
}

ImageFlow::ImageFlowAlgorithm SimpleFlow::getFlowAlgorithm() const { return ImageFlow::ImageFlowAlgorithm::SimpleFlow; }

std::string SimpleFlow::getName() { return std::string("simple"); }

Status SimpleFlow::allocMemory() {
  const int maxSize0 = (int)(mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).width() *
                             mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).height());
  const int maxSize1 = (int)(mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(0).width() *
                             mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(0).height());
  ImageFlow::allocMemory();
  FAIL_RETURN(imageLab0.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(imageLab1.alloc(maxSize1, "Optical Flow Pair"));
  FAIL_RETURN(gradient0.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(gradient1.alloc(maxSize1, "Optical Flow Pair"));

  FAIL_RETURN(confidence.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(transformedConfidence.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(flow.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(interpolatedFlow.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(upsampledFlow.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(debugInfo.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(backwardFlowImage1.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(extrapolatedBackwardFlowImage1.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(extrapolatedFlowTmp.alloc(extrapolatedFlowRects[0].getArea(), "Optical Flow Pair"));

  const int regularizedKernelSize = parameters.find(std::string("regularizedKernelSize")) != parameters.end()
                                        ? (int)parameters[std::string("regularizedKernelSize")]
                                        : 20;
  FAIL_RETURN(
      kernelWeight.alloc((2 * regularizedKernelSize + 1) * (2 * regularizedKernelSize + 1), "Optical Flow Pair"));
  return CUDA_STATUS;
}

Status SimpleFlow::findSingleScaleImageFlow(const int2& offset0, const int2& size0,
                                            const GPU::Buffer<const uint32_t>& image0, const int2& offset1,
                                            const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                            GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) {
  // Parameters used for finding flow
  const int flowSize = (int)parameters[std::string("flowSize")];
  const int windowSize = (int)parameters[std::string("windowSize")];
  const float flowMagnitudeWeight = parameters[std::string("flowMagnitudeWeight")];
  const float gradientWeight = parameters[std::string("gradientWeight")];

  // Parameters used to transform confidence value
  const float confidenceTransformThreshold = parameters[std::string("confidenceTransformThreshold")];
  const float confidenceTransformGamma = parameters[std::string("confidenceTransformGamma")];
  const float confidenceTransformClampedValue = parameters[std::string("confidenceTransformClampedValue")];

  // Parameters used to perform confidence aware interpolation
  const int interpolationKernelSize = (int)parameters[std::string("interpolationKernelSize")];
  const float interpolationSigmaSpace = parameters[std::string("interpolationSigmaSpace")];
  const float interpolationSigmaImage = parameters[std::string("interpolationSigmaImage")];
  const float interpolationSigmaConfidence = parameters[std::string("interpolationSigmaConfidence")];

  // const float interpolationSigmaTime = parameters[std::string("interpolationSigmaTime")];

  // Intermediates images in LAB color space
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size0, image0, imageLab0.borrow(), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size1, image1, imageLab1.borrow(), gpuStream));

  // Gradient of intermediate images
  FAIL_RETURN(Util::ImageProcessingGPU::findGradient(size0, imageLab0.borrow(), gradient0.borrow(), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::findGradient(size1, imageLab1.borrow(), gradient1.borrow(), gpuStream));

  // This is for my own version of simple flow
  FAIL_RETURN(findForwardFlow(flowSize, windowSize, flowMagnitudeWeight, gradientWeight, size0, offset0,
                              imageLab0.borrow_const(), gradient0.borrow_const(), size1, offset1,
                              imageLab1.borrow_const(), gradient1.borrow_const(),
                              !flowSequence ? flow.borrow() : outputFlow, confidence.borrow(), gpuStream));

  // Compute flow confidence
  FAIL_RETURN(performConfidenceTransform(size0.x, size0.y, confidenceTransformThreshold, confidenceTransformGamma,
                                         confidenceTransformClampedValue, confidence.borrow(),
                                         transformedConfidence.borrow(), gpuStream));

  // Do not generate temporal coherent flow
  if (!flowSequence) {
    // Confidence aware interpolation
    FAIL_RETURN(performConfidenceAwareFlowInterpolation(false, size0, interpolationKernelSize, interpolationSigmaSpace,
                                                        interpolationSigmaImage, interpolationSigmaConfidence,
                                                        imageLab0.borrow_const(), flow.borrow(),
                                                        transformedConfidence.borrow(), outputFlow, gpuStream));

    /*GPU::UniqueBuffer<float> frames;
    FAIL_RETURN(frames.alloc(1, "Tmp"));
    FAIL_RETURN(GPU::memsetBlocking(frames.borrow(), 0, sizeof(float)));
    FAIL_RETURN(SimpleFlow::performTemporalAwareFlowInterpolation(
      false, 0, size0,
      interpolationKernelSize, interpolationSigmaSpace, interpolationSigmaImage, interpolationSigmaTime,
      frames.borrow_const(),
      imageLab0.borrow_const(),
      flow.borrow_const(),
      transformedConfidence.borrow_const(),
      outputFlow,
      gpuStream));*/
  }
  FAIL_RETURN(gpuStream.synchronize());
  return CUDA_STATUS;
}

Status SimpleFlow::findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                           const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                           const GPU::Buffer<const uint32_t>& buffer1, GPU::Stream gpuStream) {
  FAIL_RETURN(
      findMultiScaleImageFlow(frame, level, bufferSize0, buffer0, bufferSize1, buffer1, finalFlow.borrow(), gpuStream));
  Rect rect0 = mergerPair->getBoundingInterRect(0, level);
  Rect rect1 = mergerPair->getBoundingInterRect(1, level);
  int2 size0 = make_int2((int)rect0.getWidth(), (int)rect0.getHeight());
  int2 offset0 = make_int2((int)rect0.left(), (int)rect0.top());
  int2 offset1 = make_int2((int)rect1.left(), (int)rect1.top());
  return Util::OpticalFlow::transformOffsetToFlow(size0, offset0, offset1, finalFlow.borrow(), gpuStream);
}

Status SimpleFlow::findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                           const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                           const GPU::Buffer<const uint32_t>& buffer1, GPU::Buffer<float2> outputFlow,
                                           GPU::Stream gpuStream) {
#ifdef FLOW_DEBUG_IMAGE
  {
    // Map coordinate to pixel value

    std::stringstream ss;
    ss.str("");
    ss << "inputBuffer-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "image index" << 0 << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), buffer0, bufferSize0.x, bufferSize0.y);

    ss.str("");
    ss << "inputBuffer-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "image index" << 1 << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), buffer1, bufferSize1.x, bufferSize1.y);
  }
#endif
  const int maxLevel = mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->numLevels();
  for (int i = maxLevel; i >= level; i--) {
    // Map image 0 and 1
    const LaplacianPyramid<float2>::LevelSpec<float2>& level0 =
        mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(i);
    const LaplacianPyramid<float2>::LevelSpec<float2>& level1 =
        mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(i);
    const int2 size0 = make_int2((int)level0.width(), (int)level0.height());
    const int2 offset0 = make_int2((int)mergerPair->getBoundingInterRect(0, i).left(),
                                   (int)mergerPair->getBoundingInterRect(0, i).top());
    const int2 size1 = make_int2((int)level1.width(), (int)level1.height());
    const int2 offset1 = make_int2((int)mergerPair->getBoundingInterRect(1, i).left(),
                                   (int)mergerPair->getBoundingInterRect(1, i).top());
    const int2 sizeExtrapolation =
        make_int2((int)extrapolatedFlowRects[i].getWidth(), (int)extrapolatedFlowRects[i].getHeight());
    const int2 offsetExtrapolation =
        make_int2((int)extrapolatedFlowRects[i].left(), (int)extrapolatedFlowRects[i].top());
    // Find intermediate images
    FAIL_RETURN(Util::OpticalFlow::coordLookup((int)level0.width(), (int)level0.height(), level0.data(), bufferSize0.x,
                                               bufferSize0.y, buffer0, image0.borrow(), gpuStream));

    FAIL_RETURN(Util::OpticalFlow::coordLookup((int)level1.width(), (int)level1.height(), level1.data(), bufferSize1.x,
                                               bufferSize1.y, buffer1, image1.borrow(), gpuStream));

    if (i == maxLevel) {
      // Find flow in the lowest level
      FAIL_RETURN(findSingleScaleImageFlow(
          offset0, size0, image0.borrow(), offset1, size1, image1.borrow(),
          i != level || flowSequence ? flowLaplacianPyramid->getLevel(i).data() : outputFlow, gpuStream));
      // If need to find flow sequence
      if (flowSequence) {
        // Set current (stitching) frame as the key frame for caching
        flowSequence->setKeyFrame(frame);
        // Do the caching for temporal coherent flow
        FAIL_RETURN(cacheFlowSequence(frame, i, bufferSize0, buffer0, size0, image0.borrow_const(), gpuStream));
        // Find the temporal coherent flow
        FAIL_RETURN(findTemporalCoherentFlow(
            frame, size0, i != level ? flowLaplacianPyramid->getLevel(i).data() : outputFlow, gpuStream));
        // Regularize the flow in the temporal domain
        // FAIL_RETURN(gpuStream.synchronize());
        // FAIL_RETURN(flowSequence->regularizeFlowTemporally(
        //  "coarseFlow", frame, size0, offset0, i != level ? flowLaplacianPyramid->getLevel(i).data() : outputFlow,
        //  gpuStream));

        // TODO: Need to reset the flow sequence whenever a cut or dissolve is detected
        FAIL_RETURN(flowSequence->checkForReset());
      }
      // Extrapolate flow to another set
      FAIL_RETURN(findExtrapolatedImageFlow(offset0, size0, image0.borrow_const(),
                                            i != level ? flowLaplacianPyramid->getLevel(i).data() : outputFlow, offset1,
                                            size1, image1.borrow_const(), offsetExtrapolation, sizeExtrapolation,
                                            extrapolatedFlowLaplacianPyramid->getLevel(i).data(), gpuStream));

      // if (flowSequence) {
      // gpuStream.synchronize();
      // FAIL_RETURN(flowSequence->regularizeFlowTemporally(
      //  "extrapolatedFlow", frame, sizeExtrapolation, offsetExtrapolation,
      //  extrapolatedFlowLaplacianPyramid->getLevel(i).data(), gpuStream));
      //}
    } else {
      // Perform flow up-sampling
      FAIL_RETURN(upsampleFlow(size0, offset0, image0.borrow(), size1, offset1, image1.borrow(),
                               flowLaplacianPyramid->getLevel(i + 1).data(),
                               (i != level) ? flowLaplacianPyramid->getLevel(i).data() : outputFlow, gpuStream));
      // Upsample the extrapolated flow
      FAIL_RETURN(Util::OpticalFlow::upsampleFlow22(extrapolatedFlowLaplacianPyramid->getLevel(i).data(),
                                                    extrapolatedFlowLaplacianPyramid->getLevel(i + 1).data(),
                                                    sizeExtrapolation.x, sizeExtrapolation.y, false, 16, gpuStream));
      // Multiply it by 2
      FAIL_RETURN(Util::OpticalFlow::mulFlowOperator(extrapolatedFlowLaplacianPyramid->getLevel(i).data(),
                                                     make_float2(2, 2), sizeExtrapolation.x * sizeExtrapolation.y,
                                                     gpuStream));
      // Fill in the space value of the original flow if valid
      FAIL_RETURN(Util::OpticalFlow::putOverOriginalFlow(
          offset0, size0, (i != level) ? flowLaplacianPyramid->getLevel(i).data() : outputFlow, offsetExtrapolation,
          sizeExtrapolation, extrapolatedFlowLaplacianPyramid->getLevel(i).data(), gpuStream));
    }
    FAIL_RETURN(Util::OpticalFlow::transformOffsetToFlow(
        sizeExtrapolation, offsetExtrapolation, offset1, extrapolatedFlowLaplacianPyramid->getLevel(i).data(),
        extrapolatedFlowLaplacianPyramid->getLevel(level).data(), gpuStream));

#ifdef FLOW_DEBUG_IMAGE
    if (i == level) {
      // Dump images in the intermediate space used for computing optical flow
      std::stringstream ss;
      ss.str("");
      ss << "intermediate-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << "image index" << 0
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), image0.borrow(), level0.width(), level0.height());

      ss.str("");
      ss << "intermediate-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << "image index" << 1
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), image1.borrow(), level1.width(), level1.height());

      // Dump images and their blend in the intermediate space
      Core::Rect uRect(0, 0, 0, 0), iRect(0, 0, 0, 0);
      Core::Rect::getInterAndUnion(mergerPair->getBoundingInterRect(0, 0), mergerPair->getBoundingInterRect(1, 0),
                                   iRect, uRect, mergerPair->getWrapWidth());
      GPU::UniqueBuffer<uint32_t> panoBuffer;
      GPU::UniqueBuffer<uint32_t> panoBuffer0;
      GPU::UniqueBuffer<uint32_t> panoBuffer1;
      FAIL_RETURN(panoBuffer.alloc(uRect.getArea(), "Simple Flow Tmp"));
      FAIL_RETURN(panoBuffer0.alloc(uRect.getArea(), "Simple Flow Tmp"));
      FAIL_RETURN(panoBuffer1.alloc(uRect.getArea(), "Simple Flow Tmp"));

      FAIL_RETURN(Util::ImageProcessingGPU::packBuffer<uint32_t>(
          mergerPair->getWrapWidth(), 0, mergerPair->getBoundingInterRect(0, 0), image0.borrow_const(), uRect,
          panoBuffer0.borrow(), gpuStream));
      ss.str("");
      ss << "intermediate-full-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << "image index" << 0
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer0.borrow(), uRect.getWidth(), uRect.getHeight());

      FAIL_RETURN(Util::ImageProcessingGPU::packBuffer<uint32_t>(
          mergerPair->getWrapWidth(), 0, mergerPair->getBoundingInterRect(1, 0), image1.borrow_const(), uRect,
          panoBuffer1.borrow(), gpuStream));
      FAIL_RETURN(gpuStream.synchronize());
      ss.str("");
      ss << "intermediate-full-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << "image index" << 1
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer1.borrow(), uRect.getWidth(), uRect.getHeight());

      FAIL_RETURN(Util::ImageProcessingGPU::buffer2DRGBACompactBlendOffsetOperator(
          uRect, panoBuffer.borrow(), 0.5f, uRect, panoBuffer0.borrow_const(), 0.5f, uRect, panoBuffer1.borrow_const(),
          gpuStream));
      FAIL_RETURN(gpuStream.synchronize());
      ss.str("");
      ss << "intermediate-full-blend";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << "image index" << 1
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer.borrow(), uRect.getWidth(), uRect.getHeight());

      // Dump the warped image using the computed flow
      FAIL_RETURN(Util::OpticalFlow::transformOffsetToFlow(
          size0, offset0, offset1, (i != level) ? flowLaplacianPyramid->getLevel(i).data() : outputFlow,
          flowLaplacianPyramid->getLevel(level).data(), gpuStream));

      FAIL_RETURN(Util::OpticalFlow::coordLookup((int)level0.width(), (int)level0.height(),
                                                 flowLaplacianPyramid->getLevel(level).data(), (int)level1.width(),
                                                 (int)level1.height(), image1.borrow(), image0.borrow(), gpuStream));
      ss.str("");
      ss << "flow-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), image0.borrow(), level0.width(), level0.height());

      // Dump the wraped image using the extrapolated flow
      FAIL_RETURN(Util::OpticalFlow::coordLookup(
          sizeExtrapolation.x, sizeExtrapolation.y, extrapolatedFlowLaplacianPyramid->getLevel(level).data(),
          (int)level1.width(), (int)level1.height(), image1.borrow(), extrapolatedImage1.borrow(), gpuStream));
      FAIL_RETURN(gpuStream.synchronize());
      ss.str("");
      ss << "extrapolated_flow-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), extrapolatedImage1.borrow(), sizeExtrapolation.x,
                                  sizeExtrapolation.y);

      FAIL_RETURN(Util::ImageProcessingGPU::packBuffer<uint32_t>(
          mergerPair->getWrapWidth(), 0, mergerPair->getBoundingInterRect(1, 0), extrapolatedImage1.borrow_const(),
          uRect, panoBuffer1.borrow(), gpuStream));
      FAIL_RETURN(gpuStream.synchronize());
      ss.str("");
      ss << "extrapolated_flow-full-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer1.borrow(), uRect.getWidth(), uRect.getHeight());

      FAIL_RETURN(Util::ImageProcessingGPU::buffer2DRGBACompactBlendOffsetOperator(
          uRect, panoBuffer.borrow(), 0.5f, uRect, panoBuffer0.borrow_const(), 0.5f, uRect, panoBuffer1.borrow_const(),
          gpuStream));
      FAIL_RETURN(gpuStream.synchronize());
      ss.str("");
      ss << "extrapolated_flow-full-blend-";
      ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << "level" << i << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer.borrow(), uRect.getWidth(), uRect.getHeight());
    }
#endif
    FAIL_RETURN(gpuStream.synchronize());
  }

  return Status::OK();
}

Status SimpleFlow::findTemporalCoherentFlow(const frameid_t frame, const int2& size, GPU::Buffer<float2> outputFlow,
                                            GPU::Stream gpuStream) {
  // Parameters used to perform confidence aware interpolation
  const int interpolationKernelSize = (int)parameters[std::string("interpolationKernelSize")];
  const float interpolationSigmaSpace = parameters[std::string("interpolationSigmaSpace")];
  const float interpolationSigmaImage = parameters[std::string("interpolationSigmaImage")];
  // const float interpolationSigmaConfidence = parameters[std::string("interpolationSigmaConfidence")];

  const float interpolationSigmaTime = parameters[std::string("interpolationSigmaTime")];

  TypedCached<uint32_t>* imageCache =
      dynamic_cast<TypedCached<uint32_t>*>(flowSequence->getFlowCachedBuffer("imageLab0").get());
  if (!imageCache) {
    return {Origin::ImageFlow, ErrType::ImplementationError, "Implementation Error"};
  }
  TypedCached<float>* confidenceCache =
      dynamic_cast<TypedCached<float>*>(flowSequence->getFlowCachedBuffer("confidence").get());
  if (!confidenceCache) {
    return {Origin::ImageFlow, ErrType::ImplementationError, "Implementation Error"};
  }
  TypedCached<float2>* flowCache = dynamic_cast<TypedCached<float2>*>(flowSequence->getFlowCachedBuffer("flow").get());
  if (!flowCache) {
    return {Origin::ImageFlow, ErrType::ImplementationError, "Implementation Error"};
  }
  const int frameIndex = flowSequence->getFrameIndex(frame);
  if (frameIndex < 0) {
    return {Origin::ImageFlow, ErrType::ImplementationError, "Implementation Error"};
  }

  /*FAIL_RETURN(SimpleFlow::performConfidenceAwareFlowInterpolation(
    false, size, interpolationKernelSize,
    interpolationSigmaSpace, interpolationSigmaImage, interpolationSigmaConfidence,
    imageLab0.borrow_const(), flowCache->getBuffer(), transformedConfidence.borrow(),
    outputFlow,
    gpuStream));*/

  FAIL_RETURN(performTemporalAwareFlowInterpolation(
      false, frameIndex, size, interpolationKernelSize, interpolationSigmaSpace, interpolationSigmaImage,
      interpolationSigmaTime, flowSequence->getFrames(), imageCache->getBuffer(), flowCache->getBuffer(),
      confidenceCache->getBuffer(), outputFlow, gpuStream));
  return CUDA_STATUS;
}

Status SimpleFlow::cacheFlowSequence(const frameid_t frame, const int level, const int2& /*bufferSize0*/,
                                     const GPU::Buffer<const uint32_t>& /*buffer0*/, const int2& size0,
                                     const GPU::Buffer<const uint32_t>& image0, GPU::Stream gpuStream) const {
  if (!flowSequence) {  // Temporal coherent flow is not used
    return Status::OK();
  }
  // Now do the caching properly
  int2 offset0 = make_int2((int)mergerPair->getBoundingInterRect(0, level).left(),
                           (int)mergerPair->getBoundingInterRect(0, level).top());
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size0, image0, imageLab0.borrow(), gpuStream));

  FAIL_RETURN(flowSequence->cacheBuffer<float>(frame, "confidence", size0, offset0,
                                               transformedConfidence.borrow_const(), gpuStream));
  FAIL_RETURN(flowSequence->cacheBuffer<float2>(frame, "flow", size0, offset0,
                                                flowLaplacianPyramid->getLevel(level).data(), gpuStream));
  FAIL_RETURN(
      flowSequence->cacheBuffer<uint32_t>(frame, "imageLab0", size0, offset0, imageLab0.borrow_const(), gpuStream));
  FAIL_RETURN(gpuStream.synchronize());

#ifdef FLOW_DEBUG_IMAGE
  std::stringstream ss;
  /*std::stringstream ss;
  ss.str("");
  ss << "cacheTime-";
  ss << frame << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab0.borrow_const(), size0.x, size0.y);*/

  TypedCached<uint32_t>* imageCache =
      dynamic_cast<TypedCached<uint32_t>*>(flowSequence->getFlowCachedBuffer("imageLab0").get());
  if (!imageCache) {
    return {Origin::ImageFlow, ErrType::ImplementationError, "Image cache is not implemented properly"};
  }

  std::vector<float> outFrames;
  flowSequence->getFrameIndices(outFrames);
  for (int i = 0; i < outFrames.size(); i++) {
    if (outFrames[i] >= 0) {
      ss.str("");
      ss << "cacheRead-";
      ss << outFrames[i] << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageCache->getBuffer().createSubBuffer(i * size0.x * size0.y),
                                  size0.x, size0.y);
    }
  }
#endif  // FLOW_DEBUG_IMAGE
  return CUDA_STATUS;
}

Status SimpleFlow::upsampleImageFlow(const int level, const int width0, const int height0,
                                     const GPU::Buffer<const uint32_t>& buffer0, const int width1, const int height1,
                                     const GPU::Buffer<const uint32_t>& buffer1, const int widthFlow,
                                     const int heightFlow, const GPU::Buffer<float2>& inputFlow,
                                     GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) {
  int finalLevel = -1;
  for (int i = mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->numLevels(); i >= 0; i--) {
    const LaplacianPyramid<float2>::LevelSpec<float2>& level0 =
        mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(i);
    if (level0.width() == widthFlow && level0.height() == heightFlow) {
      finalLevel = i;
      break;
    }
  }
  if (finalLevel < 0) {
    return {Origin::ImageFlow, ErrType::InvalidConfiguration, "Final level < 0"};
  }
  if (finalLevel == level) {
    GPU::memcpyBlocking(outputFlow, inputFlow.as_const(), sizeof(float2) * widthFlow * heightFlow);
    return Status::OK();
  }
  for (int i = finalLevel - 1; i >= level; i--) {
    // Map image 0 and 1
    const LaplacianPyramid<float2>::LevelSpec<float2>& level0 =
        mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(i);
    const LaplacianPyramid<float2>::LevelSpec<float2>& level1 =
        mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(i);
    const int2 size0 = make_int2((int)level0.width(), (int)level0.height());
    const int2 offset0 = make_int2((int)mergerPair->getBoundingInterRect(0, i).left(),
                                   (int)mergerPair->getBoundingInterRect(0, i).top());
    const int2 size1 = make_int2((int)level1.width(), (int)level1.height());
    const int2 offset1 = make_int2((int)mergerPair->getBoundingInterRect(1, i).left(),
                                   (int)mergerPair->getBoundingInterRect(1, i).top());

    // Map coordinate to pixel value
    FAIL_RETURN(Util::OpticalFlow::coordLookup((int)level0.width(), (int)level0.height(), level0.data(), width0,
                                               height0, buffer0, image0.borrow(), gpuStream));
    FAIL_RETURN(Util::OpticalFlow::coordLookup((int)level1.width(), (int)level1.height(), level1.data(), width1,
                                               height1, buffer1, image1.borrow(), gpuStream));

    FAIL_RETURN(performFlowUpsample22(size0, offset0, image0.borrow_const(), size1, offset1, image1.borrow_const(),
                                      i == finalLevel - 1 ? inputFlow : flowLaplacianPyramid->getLevel(i + 1).data(),
                                      flow.borrow(), i != level ? flowLaplacianPyramid->getLevel(i).data() : outputFlow,
                                      gpuStream));

    FAIL_RETURN(gpuStream.synchronize());
  }
  return CUDA_STATUS;
}

Status SimpleFlow::performFlowUpsample22(const int2& size0, const int2& offset0,
                                         const GPU::Buffer<const uint32_t>& image0, const int2& size1,
                                         const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                                         const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> tmpFlow,
                                         GPU::Buffer<float2> upsampledFlow, GPU::Stream gpuStream) {
  const int windowSize = (int)parameters[std::string("windowSize")];
  const float flowMagnitudeWeight = parameters[std::string("flowMagnitudeWeight")];
  const float gradientWeight = parameters[std::string("gradientWeight")];

  // Parameters used for flow up-sampling
  const int upsamplingJitterSize = (int)parameters[std::string("upsamplingJitterSize")];
  const float interpolationSigmaSpace = parameters[std::string("interpolationSigmaSpace")];
  const float interpolationSigmaImage = parameters[std::string("interpolationSigmaImage")];

  // Gradient of intermediate images
  FAIL_RETURN(Util::ImageProcessingGPU::findGradient(size0, image0, gradient0.borrow(), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::findGradient(size1, image1, gradient1.borrow(), gpuStream));

  // Intermediates images in LAB color space
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size0, image0, imageLab0.borrow(), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size1, image1, imageLab1.borrow(), gpuStream));
  // Upsample flow
  FAIL_RETURN(Util::OpticalFlow::upsampleFlow22(upsampledFlow, inputFlow, size0.x, size0.y, false,
                                                SIMPLE_FLOW_TILE_WIDTH, gpuStream));
  // Multiply it by 2
  FAIL_RETURN(Util::OpticalFlow::mulFlowOperator(upsampledFlow, make_float2(2, 2), size0.x * size0.y, gpuStream));
  // Now do the jittering of flow for better accuracy
  FAIL_RETURN(performFlowJittering(upsamplingJitterSize, windowSize, flowMagnitudeWeight, gradientWeight, size0,
                                   offset0, imageLab0.borrow_const(), gradient0.borrow_const(), size1, offset1,
                                   imageLab1.borrow_const(), gradient1.borrow_const(), upsampledFlow, tmpFlow,
                                   gpuStream));

  // Now perform bilateral filtering to finalize the entire process
  FAIL_RETURN(performConfidenceAwareFlowInterpolation(false, size0, 2, interpolationSigmaSpace, interpolationSigmaImage,
                                                      10, imageLab0.borrow_const(), tmpFlow, GPU::Buffer<float>(),
                                                      upsampledFlow, gpuStream));
  return Status::OK();
}

Status SimpleFlow::upsampleFlow(const int2& size0, const int2& offset0, const GPU::Buffer<const uint32_t>& image0,
                                const int2& size1, const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                                const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> upsampledFlow,
                                GPU::Stream gpuStream) {
  FAIL_RETURN(performFlowUpsample22(size0, offset0, image0, size1, offset1, image1, inputFlow, flow.borrow(),
                                    upsampledFlow, gpuStream));
  return CUDA_STATUS;
}

Status SimpleFlow::findExtrapolatedImageFlow(const int2& offset0, const int2& size0,
                                             const GPU::Buffer<const uint32_t>& image0,
                                             const GPU::Buffer<const float2>& inputFlow0, const int2& offset1,
                                             const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                             const int2& outputOffset, const int2& outputSize,
                                             GPU::Buffer<float2> outputFlow0, GPU::Stream gpuStream) {
  // const int interpolationKernelSize = (int)parameters[std::string("interpolationKernelSize")];
  const float interpolationSigmaSpace = parameters[std::string("interpolationSigmaSpace")];
  // const float interpolationSigmaImage = parameters[std::string("interpolationSigmaImage")];
  // const float interpolationSigmaConfidence = parameters[std::string("interpolationSigmaConfidence")];
  const int extrapolationKernelSize = (int)parameters[std::string("extrapolationKernelSize")];

  // Find the flow from the other image
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      size1, backwardFlowImage1.borrow(), make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      outputSize, outputFlow0, make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), gpuStream));
  FAIL_RETURN(Util::OpticalFlow::backwardCoordLookup(offset0, size0, inputFlow0, offset1, size1,
                                                     backwardFlowImage1.borrow(), gpuStream));

#ifdef FLOW_DEBUG_IMAGE
  {
    FAIL_RETURN(Util::OpticalFlow::outwardCoordLookup(offset1, size1, backwardFlowImage1.borrow_const(), offset0, size0,
                                                      image0, imageLab1.borrow(), gpuStream));

    std::stringstream ss;
    ss.str("");
    ss << "extrapolated-backward-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab1.borrow_const(), size1.x, size1.y);

    FAIL_RETURN(Util::OpticalFlow::transformOffsetToFlow(size1, offset1, offset0, backwardFlowImage1.borrow_const(),
                                                         extrapolatedFlowTmp.borrow(), gpuStream));

    FAIL_RETURN(Util::OpticalFlow::coordLookup(size1.x, size1.y, extrapolatedFlowTmp.borrow(), size0.x, size0.y, image0,
                                               imageLab1.borrow(), gpuStream));

    ss.str("");
    ss << "extrapolated-lookupbackward-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab1.borrow_const(), size1.x, size1.y);

    FAIL_RETURN(Util::OpticalFlow::transformOffsetToFlow(size0, offset0, offset1, inputFlow0,
                                                         extrapolatedFlowTmp.borrow(), gpuStream));

    FAIL_RETURN(Util::OpticalFlow::coordLookup(size0.x, size0.y, extrapolatedFlowTmp.borrow(), size1.x, size1.y, image1,
                                               imageLab1.borrow(), gpuStream));

    ss.str("");
    ss << "extrapolated-lookupforward-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab1.borrow_const(), size1.x, size1.y);
  }
#endif

  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size1, image1, imageLab1.borrow(), gpuStream));
  FAIL_RETURN(Util::ImageProcessingGPU::convertRGBToNormalizedLAB(size0, image0, imageLab0.borrow(), gpuStream));

  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      size1, extrapolatedBackwardFlowImage1.borrow(), make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), gpuStream));
  // Extrapolate flow in the other image to size1 using imageLab1 as guidance
  FAIL_RETURN(performConfidenceAwareFlowInterpolation(
      true, size1, extrapolationKernelSize, 2 * interpolationSigmaSpace, 20, 0, imageLab1.borrow_const(),
      backwardFlowImage1.borrow_const(), GPU::Buffer<float>(), extrapolatedBackwardFlowImage1.borrow(), gpuStream));
#ifdef FLOW_DEBUG_IMAGE
  {
    FAIL_RETURN(Util::OpticalFlow::outwardCoordLookup(offset1, size1, extrapolatedBackwardFlowImage1.borrow_const(),
                                                      offset0, size0, image0, imageLab1.borrow(), gpuStream));

    std::stringstream ss;
    ss.str("");
    ss << "extrapolated-backward1-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab1.borrow_const(), size1.x, size1.y);

    FAIL_RETURN(Util::OpticalFlow::transformOffsetToFlow(size1, offset1, offset0,
                                                         extrapolatedBackwardFlowImage1.borrow_const(),
                                                         extrapolatedFlowTmp.borrow(), gpuStream));

    FAIL_RETURN(Util::OpticalFlow::coordLookup(size1.x, size1.y, extrapolatedFlowTmp.borrow(), size0.x, size0.y, image0,
                                               imageLab1.borrow(), gpuStream));

    ss.str("");
    ss << "extrapolated-lookupbackward1-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), imageLab1.borrow_const(), size1.x, size1.y);
  }
#endif

  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      outputSize, extrapolatedFlowTmp.borrow(), make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), gpuStream));

  // Map the extrapolated flow back to the original domain, into a new buffer
  FAIL_RETURN(Util::OpticalFlow::forwardCoordLookup(offset1, size1, extrapolatedBackwardFlowImage1.borrow(), offset0,
                                                    size0, inputFlow0, outputOffset, outputSize,
                                                    extrapolatedFlowTmp.borrow(), gpuStream));

  FAIL_RETURN(Util::OpticalFlow::putOverOriginalFlow(offset0, size0, inputFlow0, outputOffset, outputSize,
                                                     extrapolatedFlowTmp.borrow(), gpuStream));

  // Regularize flow (by blurring) in the original image
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      outputSize, outputFlow0, make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), gpuStream));
  // Regularize flow (by blurring) in the original image
  FAIL_RETURN(performConfidenceAwareFlowInterpolation(false, outputSize, 2, 5, 0, 0, imageLab1.borrow_const(),
                                                      extrapolatedFlowTmp.borrow_const(), GPU::Buffer<float>(),
                                                      outputFlow0, gpuStream));

  return CUDA_STATUS;
}

#ifndef NDEBUG
Status SimpleFlow::dumpDebugImages(const int width0, const int height0, const GPU::Buffer<const uint32_t>& buffer0,
                                   const int width1, const int height1, const GPU::Buffer<const uint32_t>& buffer1,
                                   GPU::Stream gpuStream) const {
  // This part is to debug on the ongoing progress of warping images before merging them
  std::stringstream ss;
  ss.str("");
  ss << "input0-";
  ss << mergerPair->getImIdString(0) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), buffer0, width0, height0);

  ss.str("");
  ss << "input1-";
  ss << mergerPair->getImIdString(1) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), buffer1, width1, height1);

  // Write pano output
  GPU::UniqueBuffer<uint32_t> pano0, pano1;
  FAIL_RETURN(
      pano0.alloc(mergerPair->getBoundingPanoRect(0).getWidth() * mergerPair->getBoundingPanoRect(0).getHeight(),
                  "Tmp Image Flow"));
  FAIL_RETURN(
      pano1.alloc(mergerPair->getBoundingPanoRect(1).getWidth() * mergerPair->getBoundingPanoRect(1).getHeight(),
                  "Tmp Image Flow"));

  FAIL_RETURN(Util::OpticalFlow::coordLookup(
      (int)mergerPair->getBoundingPanoRect(0).getWidth(), (int)mergerPair->getBoundingPanoRect(0).getHeight(),
      mergerPair->getPanoToInputSpaceCoordMapping(0), width0, height0, buffer0, pano0.borrow(), gpuStream));

  FAIL_RETURN(Util::OpticalFlow::coordLookup(
      (int)mergerPair->getBoundingPanoRect(1).getWidth(), (int)mergerPair->getBoundingPanoRect(1).getHeight(),
      mergerPair->getPanoToInputSpaceCoordMapping(1), width1, height1, buffer1, pano1.borrow(), gpuStream));

  ss.str("");
  ss << "pano0-";
  ss << mergerPair->getImIdString(0) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), pano0.borrow(), mergerPair->getBoundingPanoRect(0).getWidth(),
                              mergerPair->getBoundingPanoRect(0).getHeight());

  ss.str("");
  ss << "pano1-";
  ss << mergerPair->getImIdString(1) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), pano1.borrow(), mergerPair->getBoundingPanoRect(1).getWidth(),
                              mergerPair->getBoundingPanoRect(1).getHeight());

  return Status::OK();
}
#endif

}  // namespace Core
}  // namespace VideoStitch
