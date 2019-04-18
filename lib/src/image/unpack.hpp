// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "colorArray.hpp"

#include "gpu/2dBuffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/frame.hpp"

#include <stdint.h>
#include <cstring>

namespace VideoStitch {
namespace Image {
/**
 * Switch on the format to do the input conversion.
 */
Status unpackCommonPixelFormat(PixelFormat format, GPU::Surface& dst, GPU::Buffer<const unsigned char> src,
                               std::size_t width, std::size_t height, GPU::Stream stream);

// ------------------------------------------------------------------------------------------------------

/**
 * Copy RGBA to pitched memory.
 * Width/height in pixels.
 */
Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& input, size_t width, size_t height,
                  GPU::Stream s);
Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Surface& input, size_t width, size_t height, GPU::Stream s);

/**
 * Copy single channel 32-bits float to pitched memory.
 * Width/height in pixels.
 */

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& input, size_t width, size_t height,
                   GPU::Stream s);

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Surface& input, size_t width, size_t height, GPU::Stream s);

/**
 * Encode single channel 32-bits float to single channel unsigned short with a scaling factor of 1000
 * Width/height in pixels.
 */
Status unpackGrayscale16(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& input, size_t width, size_t height,
                         GPU::Stream s);

Status unpackGrayscale16(GPU::Buffer2D& dst, const GPU::Surface& input, size_t width, size_t height, GPU::Stream s);

/**
 * Encode single channel 32-bits float to YV12.
 * Width/height in pixels.
 */

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                   const GPU::Buffer<const uint32_t>& input, size_t width, size_t height, GPU::Stream s);

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& input,
                   size_t width, size_t height, GPU::Stream s);

/**
 * Clip RGBA to RGB.
 * Width/height in pixels.
 */
Status unpackRGB(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& input, size_t width, size_t height,
                 GPU::Stream s);
Status unpackRGB(GPU::Buffer2D& dst, const GPU::Surface& input, size_t width, size_t height, GPU::Stream s);

/**
 * Convert RGBA to planar 12bits 4:2:0 (YV12).
 * Width/height in pixels.
 */
Status unpackYV12(GPU::Buffer2D& dst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                  const GPU::Buffer<const uint32_t>& input, size_t width, size_t height, GPU::Stream s);
Status unpackYV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& input,
                  size_t width, size_t height, GPU::Stream s);

/**
 * Convert RGBA to interleaved 12bits 4:2:0 (NV12).
 * Width/height in pixels.
 */
Status unpackNV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uvDst, const GPU::Buffer<const uint32_t>& input, size_t width,
                  size_t height, GPU::Stream s);
Status unpackNV12(GPU::Buffer2D& dst, GPU::Buffer2D& uvDst, const GPU::Surface& input, size_t width, size_t height,
                  GPU::Stream s);

/**
 * Convert RGBA to YUV422 with YUY2 components ordering.
 * Width/height in pixels.
 */
Status unpackYUY2(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                  GPU::Stream s);
Status unpackYUY2(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height, GPU::Stream s);

/**
 * Convert RGBA to YUV422 with UYVY components ordering.
 * Width/height in pixels.
 */
Status unpackUYVY(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                  GPU::Stream s);
Status unpackUYVY(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height, GPU::Stream s);

/**
 * Convert RGBA to YUV422P10 (10-bits planar YUV 4:2:2).
 * Width/height in pixels.
 */
Status unpackYUV422P10(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                       const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height, GPU::Stream s);
Status unpackYUV422P10(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& src,
                       std::size_t width, std::size_t height, GPU::Stream s);

/**
 * Convert RGBA to Grayscale.
 * Width/height in pixels.
 */
Status unpackGrayscale(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                       GPU::Stream s);

// ------------------------------------------------------------------------------------------------------

/**
 * Convert RGB to RGBA out of place in cuda stream s
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers
 * @param height of both buffers
 * @param blockSize cuda block size.
 * @param stream cuda stream
 */
Status convertRGBToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                        GPU::Stream s);

/**
 * Convert RGB to RGBA out of place in cuda stream s
 * Debug use.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers
 * @param height of both buffers
 * @param stream cuda stream
 */
Status convertRGB210ToRGBA(GPU::Surface& dst, GPU::Buffer<const uint32_t> src, std::size_t width, std::size_t height,
                           GPU::Stream s);

/**
 * Convert BGR888 to RGBA out of place in cuda stream s
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers
 * @param height of both buffers
 * @param blockSize cuda block size.
 * @param stream cuda stream
 */
Status convertBGRToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                        std::size_t height, GPU::Stream s);

/**
 * Convert BGRU8888 (where 'U' stands for 'unused') to RGB888 out of place in cuda stream s
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers
 * @param height of both buffers
 * @param blockSize cuda block size.
 * @param stream cuda stream
 */
Status convertBGRUToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                         std::size_t height, GPU::Stream s);

/**
 * Convert Bayer-filtered pattern:
 * R G
 * G B
 * to RGB888 out of place in cuda stream s
 * @param dst Destination buffer. Should be of size 4 * width * height.
 * @param src Source buffer.
 * @param width of @a src
 * @param height of @a src
 * @param blockSize cuda block size.
 * @param s cuda stream
 */
Status convertBayerRGGBToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);

/**
 * Convert Bayer-filtered pattern:
 * B G
 * G R
 * to RGB888 out of place in cuda stream s
 * @param dst Destination buffer. Should be of size 4 * width * height.
 * @param src Source buffer.
 * @param width of @a src
 * @param height of @a src
 * @param blockSize cuda block size.
 * @param s cuda stream
 */
Status convertBayerBGGRToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);

/**
 * Convert Bayer-filtered pattern:
 * G R
 * B G
 * to RGB888 out of place in cuda stream s
 * @param dst Destination buffer. Should be of size 4 * width * height.
 * @param src Source buffer.
 * @param width of @a src
 * @param height of @a src
 * @param blockSize cuda block size.
 * @param s cuda stream
 */
Status convertBayerGRBGToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);

/**
 * Convert Bayer-filtered pattern:
 * G B
 * R G
 * to RGB888 out of place in cuda stream s
 * @param dst Destination buffer. Should be of size 4 * width * height.
 * @param src Source buffer.
 * @param width of @a src
 * @param height of @a src
 * @param blockSize cuda block size.
 * @param s cuda stream
 */
Status convertBayerGBRGToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);

/**
 * Convert planar 12 bits 4:2:0 (YV12) to RGBA8888 out of place in cuda stream s.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers. Must be even.
 * @param s cuda stream
 */
Status convertYV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream s);

/**
 * Convert interleaved 12 bits 4:2:0 (NV12) to RGBA8888 out of place in cuda stream s.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers. Must be even.
 * @param s cuda stream
 */
Status convertNV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream s);

/**
 * Convert 10bpp YUV422 planar to RGBA out of place in cuda stream s.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers. Must be even.
 * @param s cuda stream
 */
Status convertYUV422P10ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);

/**
 * Convert YUV422 (8bits) to RGBA out of place in cuda stream s.
 * Order is 'U Y0 V Y1' (two pixels at a time).
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers Must be even.
 */
Status convertUYVYToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream s);

/**
 * Convert YUV422 (8bits) to RGBA out of place in cuda stream s.
 * Order is 'Y0 U Y1 V' (two pixels at a time).
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers Must be even.
 * @param s cuda stream
 */
Status convertYUY2ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream s);

/**
 * Convert Grayscale (mono 8bits) to RGBA out of place in cuda stream s.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param width of both buffers. Must be even.
 * @param height of both buffers Must be even.
 * @param s cuda stream
 */
Status convertGrayscaleToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream s);
}  // namespace Image
}  // namespace VideoStitch
