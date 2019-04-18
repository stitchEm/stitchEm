// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMAGEOPS_HPP_
#define IMAGEOPS_HPP_

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include <stdint.h>

namespace VideoStitch {
namespace Image {

/**
 * Piecewise MUL operator: dst = dst & toMul.
 **/
template <typename T>
Status mulOperatorRaw(GPU::Buffer<T> dst, const T toMul, std::size_t size, GPU::Stream stream);

/**
 * Piecewise AND operator: dst = dst & toAnd.
 **/
template <typename T>
Status andOperatorRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAnd, std::size_t size, GPU::Stream stream);

/**
 * Subtract an image from another: dst -= toSubtract.
 * Alpha is handled the following way:
 *  a   b   (a - b)
 *  1   1      1       normal case
 *  0   0      0       obviously
 *  1   0      1       value is a
 *  0   1      0       This is the hard part. I chose 0. value is unimportant.
 * @param dst Left hand side.
 * @param toSubtract Right hand side
 * @param size Size in pixels of both images.
 * @param stream CUDA stream for the operation
 */
template <typename T>
Status subtractRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toSubtract, std::size_t size, GPU::Stream stream);
// Expects 8-bit RGBA pixels as input, outputs 10-bit RGBA
Status subtract(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                GPU::Stream stream);

/**
 * Add an image to another: dst += toAdd.
 * Alpha is handled the following way:
 *  a   b   (a + b)
 *  1   1      1       normal case
 *  0   0      0       obviously
 *  1   0      1       value is a
 *  0   1      0       value is unimportant.
 *
 * WARNING: it means that addition is non-commutative!
 * @param dst Left hand side.
 * @param toAdd Right hand side
 * @param size Size in pixels of both images.
 * @param stream CUDA stream for the operation
 */

// ----- Scalars
template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd, std::size_t size, GPU::Stream stream);

// ----- Pixels
// Expect 10-bit RGBA for first image, 8-bit RGBA for second; result in 10-bit RGBA
Status add10n8(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream);
// Expect 10-bit RGBA for first image, 8-bit RGBA for second; result clamped
Status add10n8Clamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream);
// Works in 10-bit RGBA domain
Status add10(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream);
// Expects 10-bit RGBA, clamp the result to 8-bit RGBA
Status addClamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream);

/**
 * Add an image to another: dst = toAdd0 + toAdd1.
 * Alpha is handled the following way:
 *  a   b   (a + b)
 *  1   1      1       normal case
 *  0   0      0       obviously
 *  1   0      1       value is a
 *  0   1      0       value is unimportant.
 *
 * WARNING: it means that addition is non-commutative!
 * @param dst Output result.
 * @param toAdd0 The first buffer.
 * @param toAdd1 The second buffer.
 * @param size Size in pixels of both images.
 * @param stream CUDA stream for the operation
 */
template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd0, GPU::Buffer<const T> toAdd1, std::size_t size,
              GPU::Stream stream);
Status addRGB210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd0, GPU::Buffer<const uint32_t> toAdd1,
                 std::size_t size, GPU::Stream stream);

/**
 * Note:
 *  a   b     (a - b) + b
 *  1   1         1       value is a
 *  0   0         0       obviously
 *  1   0         1       value is a
 *  0   1         0       consistent.
 */

Status rotate(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream);
Status rotateLeft(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream);
Status rotateRight(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream);
Status flip(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream);
Status flop(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream);

}  // namespace Image
}  // namespace VideoStitch

#endif
