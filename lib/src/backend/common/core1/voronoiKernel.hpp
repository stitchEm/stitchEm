// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/panoDimensions.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace Core {

Status voronoiInit(GPU::Buffer<uint32_t> buffer, std::size_t width, std::size_t height, uint32_t blackMask,
                   uint32_t whiteMask, unsigned blockSize, GPU::Stream stream);

Status voronoiComputeEuclidean(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                               std::size_t height, uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream);

Status voronoiComputeErect(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, const PanoRegion& region,
                           uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream);

Status voronoiMakeMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width, std::size_t height,
                       unsigned blockSize, GPU::Stream stream);

Status initForMaskComputation(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                              std::size_t height, uint32_t mask, uint32_t otherMask, GPU::Stream stream);

Status makeMaskErect(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                     GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                     float maxTransitionDistance, float power, GPU::Stream stream);

Status makeMaskEuclidean(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                         GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                         float maxTransitionDistance, float power, GPU::Stream stream);

Status extractEuclideanDist(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> whiteResult, std::size_t width,
                            std::size_t height, bool hWrap, float maxTransitionDistance, float power,
                            GPU::Stream stream);

}  // namespace Core
}  // namespace VideoStitch
