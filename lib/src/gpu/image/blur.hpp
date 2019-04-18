// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BLUR_HPP_
#define BLUR_HPP_

#include <stdint.h>
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {
namespace Image {

Status gaussianBlur2D(GPU::Buffer<unsigned char> buf, GPU::Buffer<unsigned char> work, std::size_t width,
                      std::size_t height, unsigned radius, unsigned passes, bool wrap, unsigned blockSize,
                      GPU::Stream stream);

/**
 * Blur @buf with a gaussian filter of radius @radius. @work must be at least as big as @buf.
 * @passes is the number of box filtering passes and must be even (performance reasons).
 */
template <typename T>
Status gaussianBlur2D(GPU::Buffer<T> dst, GPU::Buffer<const T> src, GPU::Buffer<T> work, std::size_t width,
                      std::size_t height, unsigned radius, unsigned passes, bool wrap, GPU::Stream stream);

/**
 * Specialized gaussian blur:
 *  - Applies to an RGBA formatted buffer, colors are blurred independently.
 *  - Output is written to dst.
 * Passes need not be even nor odd.
 */
Status gaussianBlur2DRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work,
                          std::size_t width, std::size_t height, unsigned radius, unsigned passes, bool wrap,
                          GPU::Stream stream);
Status gaussianBlur2DRGB210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work,
                            std::size_t width, std::size_t height, unsigned radius, unsigned passes, bool wrap,
                            GPU::Stream stream);

/**
 * Small-support optimized version.
 * In-place.
 */
Status gaussianBlur2DRGBASS(GPU::Buffer<uint32_t> buf, GPU::Buffer<uint32_t> work, std::size_t width,
                            std::size_t height, std::size_t radius, bool wrap, GPU::Stream stream);
Status gaussianBlur2DRGB210SS(GPU::Buffer<uint32_t> buf, GPU::Buffer<uint32_t> work, std::size_t width,
                              std::size_t height, std::size_t radius, bool wrap, GPU::Stream stream);

template <typename T>
Status boxBlur1DNoWrap(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t width, std::size_t height,
                       unsigned radius, unsigned blockSize, GPU::Stream stream);
Status boxBlur1DNoWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                              std::size_t height, unsigned radius, GPU::Stream stream);

Status boxBlur1DRowWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                               std::size_t height, unsigned radius, GPU::Stream stream);

template <typename T>
Status boxBlur1DWrap(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t width, std::size_t height,
                     unsigned radius, unsigned blockSize, GPU::Stream stream);

Status boxBlurColumnsNoWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                                   std::size_t height, unsigned radius, GPU::Stream gpuStream);
Status boxBlurColumnsWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                                 std::size_t height, unsigned radius, GPU::Stream gpuStream);

Status boxBlurRowsRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                          std::size_t height, unsigned radius, GPU::Stream stream, bool wrap);

Status boxBlur1DWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                            std::size_t height, unsigned radius, GPU::Stream stream);
Status gaussianBlur1DRGBA210SS(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                               std::size_t height, unsigned radius, bool wrap, GPU::Stream stream);

}  // namespace Image
}  // namespace VideoStitch

#endif
