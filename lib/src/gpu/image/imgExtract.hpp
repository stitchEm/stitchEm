// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMGEXTRACT_HPP_
#define IMGEXTRACT_HPP_

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {
namespace Image {
/**
 * Extract a portion of an image.
 * @param dst Destination
 * @param dstWidth Destination width
 * @param dstHeight Destination height
 * @param src Source
 * @param srcWidth Source width
 * @param srcHeight Source height
 * @param offsetX x offset of dst in src.
 * @param offsetY y offset of dst in src.
 * @param hWrap If true, src is considered to wrap horizontally.
 * @param blockSize The size of the cuda block (blockSize x blockSize)
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
Status imgExtractFrom(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                      GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                      std::size_t offsetY, bool hWrap, GPU::Stream stream);
}  // namespace Image
}  // namespace VideoStitch
#endif
