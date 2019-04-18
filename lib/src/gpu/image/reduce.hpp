// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/util.hpp"
#include "libvideostitch/status.hpp"
#include "../util.hpp"

#include <stdint.h>
#include <cstring>

namespace VideoStitch {
namespace Image {

namespace {
// Read the article if you want to change this.
static const int BLOCKSIZE = 256;
}  // namespace

inline std::size_t getReduceWorkBufferSize(std::size_t size) {
  std::size_t result = 0;
  while (size > 1) {
    const std::size_t dstBlocks = ceilDiv(size, 2 * BLOCKSIZE);
    result += dstBlocks;
    size = dstBlocks;
  }
  return result;
}

/**
 * Reduce-sum the given buffer.
 * @param src Source buffer.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceSum(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size, uint32_t& result);

/**
 * Reduce-sum the given RGBA210 buffer. Only solid pixels are considered.
 * @param src Source buffer.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceSumSolid(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size, uint32_t& result);

/**
 * Reduce-count the number of solid pixels.
 * @param src Source buffer RGBA210 format.
 * @param work Work buffer. Must be of size at least getReduceWorkBufferSize(@a size).
 * @param size of @a src
 * @param result On success, will contain the result.
 * @note Synchronous
 */
Status reduceCountSolid(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size,
                        uint32_t& result);
}  // namespace Image
}  // namespace VideoStitch
