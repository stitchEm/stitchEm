// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./imageWarper.hpp"

#include "./linearFlowWarper.hpp"
#include "./noWarper.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageWarper> ImageWarper::factor(const ImageWarperAlgorithm e, std::shared_ptr<MergerPair> mergerPair,
                                           std::map<std::string, float> parameters, GPU::Stream gpuStream) {
  std::unique_ptr<ImageWarper> warperPtr;
  switch (e) {
#ifndef VS_OPENCL
    case ImageWarperAlgorithm::LinearFlowWarper: {
      warperPtr.reset(new LinearFlowWarper(parameters));
    } break;
#endif
    case ImageWarperAlgorithm::NoWarper:
      warperPtr.reset(new NoWarper(parameters));
      break;
    default:
      warperPtr.reset(new NoWarper(parameters));
      break;
  }
  FAIL_RETURN(warperPtr->init(mergerPair));
  FAIL_RETURN(warperPtr->setupCommon(gpuStream));
  return Potential<ImageWarper>(warperPtr.release());
}

ImageWarper::ImageWarperAlgorithm ImageWarper::getDefaultImageWarper() { return ImageWarperAlgorithm::NoWarper; }

ImageWarper::ImageWarper(const std::map<std::string, float>& parameters_)
    : parameters(parameters_), mergerPair(nullptr) {}

ImageWarper::~ImageWarper() {}

Status ImageWarper::init(std::shared_ptr<MergerPair>& inMergerPair) {
  this->mergerPair = inMergerPair;
  return Status::OK();
}

Status ImageWarper::warp(GPU::Buffer<uint32_t> warpedBuffer, const GPU::Buffer<const uint32_t> inputBuffer, const Rect&,
                         const GPU::Buffer<const float2> flow, const int, const int, GPU::Buffer<float4> debug,
                         GPU::Buffer<uint32_t> flowWarpedBuffer, GPU::Stream gpuStream) {
  return Status::OK();
}

Status ImageWarper::setupCommon(GPU::Stream gpuStream) { return Status::OK(); }

}  // namespace Core
}  // namespace VideoStitch
