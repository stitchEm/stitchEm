// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SAMPLING_HPP_
#define SAMPLING_HPP_

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include <stdint.h>

namespace VideoStitch {

namespace GPU {
class Surface;
}

namespace Image {

// ----------------------- Subsampling -------------------

/**
 * Subsample a buffer by a factor of two, use bilinear interpolation in the local neighborhood of size 2x2
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status subsample22(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t srcWidth, std::size_t srcHeight,
                   GPU::Stream stream);

/**
 * Subsample a buffer by a factor of two, picking the top left value for every 2x2 pixels blocks.
 * WARNING: no antialiasing filter !
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param blockSize Cuda block size (effective size is blockSize * blockSize)
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status subsample22Nearest(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t srcWidth, std::size_t srcHeight,
                          unsigned int blockSize, GPU::Stream stream);

/**
 * Subsample a buffer by a factor of two, picking the topleft value for every 2x2 pixels blocks.
 * WARNING: no antialiasing filter !
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param dstMask subsampled buffer weight, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcMask subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param blockSize Cuda block size (effective size is blockSize * blockSize)
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status subsample22Mask(GPU::Buffer<T> dst, GPU::Buffer<uint32_t> dstMask, GPU::Buffer<const T> src,
                       GPU::Buffer<const uint32_t> srcMask, std::size_t srcWidth, std::size_t srcHeight,
                       unsigned int blockSize, GPU::Stream stream);

/**
 * Subsamples the given mask by a factor of two. For each 2x2 pixel block, the output pixel is masked out if any of the
 * input pixels is masked out (i.e. any pixel has value 1).
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param blockSize Cuda block size (effective size is blockSize * blockSize)
 * @param stream Cuda stream to run in.
 */
Status subsampleMask22(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t srcWidth,
                       std::size_t srcHeight, unsigned int blockSize, GPU::Stream stream);

/**
 * Subsamples an image. Whenever a group of 4 pixels has all of its pixels with 0 alpha, the resulting pixel has 0
 * alpha. Else, the resulting pixel is the average of the solid pixels.
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param stream Cuda stream to run in.
 */
Status subsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t srcWidth,
                       std::size_t srcHeight, GPU::Stream stream);

// ----------------------- Upsampling -------------------

/**
 * Upsamples a buffer.
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param dstWeight subsampled buffer weight, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status upsample22(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t dstWidth, std::size_t dstHeight, bool wrap,
                  GPU::Stream stream);

/**
 * Upsamples an image in RGB210.
 * @param dst subsampled buffer, size dstWidth * dstHeight.
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param stream Cuda stream to run in.
 */
Status upsample22RGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                         std::size_t dstHeight, bool wrap, GPU::Stream stream);

/**
 * Upsamples an image in RGBA.
 * @param dst subsampled buffer, size dstWidth * dstHeight.
 * @param dst subsampled buffer, size ((srcWidth + 1) / 2) * ((srcHeight + 1) / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param stream Cuda stream to run in.
 */
Status upsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                      std::size_t dstHeight, bool wrap, GPU::Stream stream);

}  // namespace Image
}  // namespace VideoStitch

#endif
