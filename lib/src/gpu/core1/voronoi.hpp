// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/panoDimensions.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {
/**
 * Compute the generalized voronoi diagram of @a src with euclidean distance
 * @param dst Output buffer for the voronoi diagram. Only two values: 0 and 255.
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param work A work buffer.
 * @param width Width of the previous buffers.
 * @param height Height of the previous buffers.
 * @param fromIdMask Bit mask of the first input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param toIdMask Bit mask of the second input.
 * @param hWrap If true, we consider the buffer to wrap horizontally.
 * @param blockSize GPU block size
 * @param stream GPU stream where to run the kernels.
 */
Status voronoiCompute(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work,
                      std::size_t width, std::size_t height, uint32_t fromIdMask, uint32_t toIdMask, bool hWrap,
                      unsigned blockSize, GPU::Stream stream);

/**
 * Compute a blending mask between two image regions
 *
 * Variation of a of 2-site generalized voronoi diagram computation using Jump Flooding,
 * modified to enable creating smooth transition maps.
 *
 * We first compute two distance maps to each of the images, then merge them.
 *
 * @param dst Output buffer for the mask. Output values are in [0;255].
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param work1 First work buffer. Same size as @a src.
 * @param work2 Second work buffer. Same size as @a src.
 * @param proj Output projection of the pano. Uses angular distance for equirect, euclidean distance otherwise.
 * @param region Output rect that describes the region of the source image in the pano
 * @param fromIdMask Bit mask of the first input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param toIdMask Bit mask of the second input.
 * @param hWrap If true, we consider the buffer to wrap horizontally.
 * @param maxTransitionDistance maximum width of the transition / overlap. In rad for equirect pano, otherwise in output
 * pixels.
 * @param power parameter of the p-norm that's used to calculate the transition. Should be >= 2.0 to use at least L2.
 * Steeper transition with larger power.
 * @param stream GPU stream where to run the kernels.
 */
Status computeMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work1,
                   GPU::Buffer<uint32_t> work2, const PanoRegion& region, uint32_t fromIdMask, uint32_t toIdMask,
                   bool hWrap, float maxTransitionDistance, float power, GPU::Stream stream);

/**
 * Set the blending mask for the first image
 * @param dst Output buffer for the mask. Output values are in [0;255].
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param fromIdMask Bit mask of the first input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param stream GPU stream where to run the kernels.
 */
Status setInitialImageMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                           std::size_t height, uint32_t fromIdMask, GPU::Stream stream);

/**
 * Compute the euclidean distance map around a source
 * For the parameters, see the computeMask function
 */
Status computeEuclideanDistanceMap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const uint32_t> src,
                                   GPU::Buffer<uint32_t> work1, GPU::Buffer<uint32_t> work2, std::size_t width,
                                   std::size_t height, uint32_t idMask, bool hWrap, float maxTransitionDistance,
                                   float power, GPU::Stream stream);

/**
 * Compute the euclidean distance map between two sources
 * For the parameters, see the computeMask function
 */
Status computeEuclideanDistanceMap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const uint32_t> src,
                                   GPU::Buffer<uint32_t> work1, GPU::Buffer<uint32_t> work2, std::size_t width,
                                   std::size_t height, uint32_t fromIdMask, uint32_t toIdMask, bool hWrap,
                                   float maxTransitionDistance, float power, GPU::Stream stream);

}  // namespace Core
}  // namespace VideoStitch
