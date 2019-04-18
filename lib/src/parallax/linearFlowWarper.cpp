// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./linearFlowWarper.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "core1/imageMerger.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "gpu/core1/voronoi.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "parse/json.hpp"

#include "libvideostitch/parse.hpp"

//#define WARPER_DEBUG

#ifdef WARPER_DEBUG
#ifndef NDEBUG
#include "util/debugUtils.hpp"
#include <sstream>
#endif
#endif

namespace VideoStitch {
namespace Core {

#define LINEAR_FLOW_WRAPER_MAX_TRANSITION_DISTANCE 150
#define LINEAR_FLOW_WRAPER_POWER 1.0

Potential<ImageWarperFactory> LinearFlowWarper::Factory::parse(const Ptv::Value& value) {
  int maxTransitionDistance = LINEAR_FLOW_WRAPER_MAX_TRANSITION_DISTANCE;
  if (Parse::populateInt("LinearFlowWarperFactory", value, "maxTransitionDistance", maxTransitionDistance, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'maxTransitionDistance' configuration, expected int"};
  }
  double power = LINEAR_FLOW_WRAPER_POWER;
  if (Parse::populateDouble("LinearFlowWarperFactory", value, "power", power, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'maxTransitionDistance' configuration, expected double"};
  }
  return Potential<ImageWarperFactory>(new LinearFlowWarper::Factory((float)maxTransitionDistance, power));
}

LinearFlowWarper::Factory::Factory(const float maxTransitionDistance, const double power)
    : maxTransitionDistance(maxTransitionDistance), power(power) {}

Ptv::Value* LinearFlowWarper::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue(LinearFlowWarper::getName()));
  res->push("maxTransitionDistance", new Parse::JsonValue(maxTransitionDistance));
  res->push("power", new Parse::JsonValue(power));
  return res;
}

std::string LinearFlowWarper::Factory::getImageWarperName() const { return LinearFlowWarper::getName(); }

std::string LinearFlowWarper::Factory::hash() const {
  std::stringstream ss;
  ss << "LinearFlowWarper "
     << "maxTransitionDistance " << maxTransitionDistance << "power " << power;
  return ss.str();
}

Potential<ImageWarper> LinearFlowWarper::Factory::create() const {
  std::map<std::string, float> parameters;
  parameters["maxTransitionDistance"] = (float)maxTransitionDistance;
  parameters["power"] = (float)power;
  return Potential<ImageWarper>(new LinearFlowWarper(parameters));
}

bool LinearFlowWarper::Factory::needsInputPreProcessing() const { return true; }

ImageWarperFactory* LinearFlowWarper::Factory::clone() const { return new Factory(maxTransitionDistance, power); }

LinearFlowWarper::LinearFlowWarper(const std::map<std::string, float>& parameters_) : ImageWarper(parameters_) {
  if (parameters.find("maxTransitionDistance") == parameters.end()) {
    parameters["maxTransitionDistance"] = LINEAR_FLOW_WRAPER_MAX_TRANSITION_DISTANCE;
  }
  if (parameters.find("power") == parameters.end()) {
    parameters["power"] = LINEAR_FLOW_WRAPER_POWER;
  }
}

std::string LinearFlowWarper::getName() { return std::string("linearflow"); }

const GPU::Buffer<const unsigned char> LinearFlowWarper::getLinearMaskWeight() const {
  return linearMaskWeight.borrow_const();
}

Rect LinearFlowWarper::getMaskRect() const { return mergerPair->getBoundingPanosIRect(); }

bool LinearFlowWarper::needImageFlow() const { return true; }

Status LinearFlowWarper::setupCommon(GPU::Stream gpuStream) {
  if (!mergerPair->doesOverlap()) {
    return Status::OK();
  }
  const float maxTransitionDistance = parameters["maxTransitionDistance"];
  const double power = parameters["power"];
  // Use Voronoi mask generated from voronoiKernel to generate a smooth transition
  // from the first to the second image
  // Prepare our own devMask
  Rect iRect = mergerPair->getBoundingPanosIRect();
  GPU::UniqueBuffer<uint32_t> work1;
  GPU::UniqueBuffer<uint32_t> work2;
  GPU::UniqueBuffer<uint32_t> devMask;
  FAIL_RETURN(devMask.alloc(iRect.getWidth() * iRect.getHeight(), "Linear Flow Warper"));
  FAIL_RETURN(work1.alloc(iRect.getWidth() * iRect.getHeight(), "Linear Flow Warper"));
  FAIL_RETURN(work2.alloc(iRect.getWidth() * iRect.getHeight(), "Linear Flow Warper"));
  FAIL_RETURN(linearMaskWeight.alloc(iRect.getWidth() * iRect.getHeight(), "Linear Flow Warper"));
  FAIL_RETURN(mergerPair->setupPairMappingMask(devMask.borrow(), gpuStream));
#ifdef WARPER_DEBUG
  {
    std::stringstream ss;
    ss.str("");
    ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/flow/";
    ss << "panoToInput0-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBACoordinateDeviceBuffer(ss.str().c_str(), mergerPair->getPanoToInputSpaceCoordMapping(0),
                                          mergerPair->getBoundingPanoRect(0).getWidth(),
                                          mergerPair->getBoundingPanoRect(0).getHeight());
  }
  {
    std::stringstream ss;
    ss.str("");
    ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/flow/";
    ss << "warperMaskDev-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), devMask.borrow_const(), iRect.getWidth(), iRect.getHeight());
  }
#endif
  FAIL_RETURN(computeEuclideanDistanceMap(linearMaskWeight.borrow(), devMask.borrow(), work1.borrow(), work2.borrow(),
                                          iRect.getWidth(), iRect.getHeight(), 1 << 2, 1 << 1, false,
                                          maxTransitionDistance, (float)power, gpuStream));

#ifdef WARPER_DEBUG
  {
    std::stringstream ss;
    ss.str("");
    ss << "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/data/flow/";
    ss << "warperMaskWeight-";
    ss << mergerPair->getImIdString(0) << " - " << mergerPair->getImIdString(1) << ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), linearMaskWeight.borrow_const(),
                                                     iRect.getWidth(), iRect.getHeight());
  }
#endif
  return CUDA_STATUS;
}

ImageWarper::ImageWarperAlgorithm LinearFlowWarper::getWarperAlgorithm() const {
  return ImageWarper::ImageWarperAlgorithm::LinearFlowWarper;
}

}  // namespace Core
}  // namespace VideoStitch
