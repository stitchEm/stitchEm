// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/strip.hpp"

namespace VideoStitch {
namespace Core {

/**
 * Mask everything except a vertical strip in warped image space.
 * @param dst Destination
 * @param dstWidth Destination width
 * @param dstHeight Destination height
 * @param geoParams The parameters defining the warp to spherical space
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
Status vStrip(GPU::Buffer<unsigned char> /*dst*/, std::size_t /*dstWidth*/, std::size_t /*dstHeight*/, float /*min*/,
              float /*max*/, InputDefinition::Format /*fmt*/, float /*distCenterX*/, float /*distCenterY*/,
              const TransformGeoParams& /*geoParams*/, const float2& /*inputScale*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Vertical strip mask not implemented in OpenCL backend"};
}

/**
 * Mask everything except a horizontal strip in warped image space.
 * @param dst Destination
 * @param dstWidth Destination width
 * @param dstHeight Destination height
 * @param geoParams The parameters defining the warp to spherical space
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
Status hStrip(GPU::Buffer<unsigned char> /*dst*/, std::size_t /*dstWidth*/, std::size_t /*dstHeight*/, float /*min*/,
              float /*max*/, InputDefinition::Format /*fmt*/, float /*distCenterX*/, float /*distCenterY*/,
              const TransformGeoParams& /*geoParams*/, const float2& /*inputScale*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Horizonal strip mask not implemented in OpenCL backend"};
}

}  // namespace Core
}  // namespace VideoStitch
