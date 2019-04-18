// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/reduce.hpp"

namespace VideoStitch {
namespace Image {

/**
 * Reduce-sum the given buffer.
 * @param src Source buffer.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceSum(GPU::Buffer<const uint32_t> /*src*/, GPU::Buffer<uint32_t> /*work*/, std::size_t /*size*/,
                 uint32_t& /*result*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "Reduce-sum not implemented in OpenCL backend"};
}

/**
 * Reduce-sum the given RGBA210 buffer. Only solid pixels are considered.
 * @param src Source buffer.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceSumSolid(GPU::Buffer<const uint32_t> /*src*/, GPU::Buffer<uint32_t> /*work*/, std::size_t /*size*/,
                      uint32_t& /*result*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "Reduce-sum not implemented in OpenCL backend for RGBA210"};
}

/**
 * Reduce-count the number of solid pixels.
 * @param src Source buffer RGBA210 format.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceCountSolid(GPU::Buffer<const uint32_t> /*src*/, GPU::Buffer<uint32_t> /*work*/, std::size_t /*size*/,
                        uint32_t& /*result*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "Reduce-count not implemented in OpenCL backend"};
}

}  // namespace Image
}  // namespace VideoStitch
