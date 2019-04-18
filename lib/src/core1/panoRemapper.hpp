// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/allocator.hpp"

#include "core/rect.hpp"

namespace VideoStitch {
namespace Core {

Status reprojectAlphaToCubemap(int panoWidth, int panoHeight, int faceLength, GPU::Surface&, Rect equirectBB,
                               GPU::Buffer<unsigned char> xPosAlpha, Rect xPosBB, GPU::Buffer<unsigned char> xNegAlpha,
                               Rect xNegBB, GPU::Buffer<unsigned char> yPosAlpha, Rect yPosBB,
                               GPU::Buffer<unsigned char> yNegAlpha, Rect yNegBB, GPU::Buffer<unsigned char> zPosAlpha,
                               Rect zPosBB, GPU::Buffer<unsigned char> zNegAlpha, Rect zNegBB, bool equiangular,
                               GPU::Stream);

Status rotateCubemap(const PanoDefinition& pano, GPU::CubemapSurface& cubemapSurface, GPU::Buffer<uint32_t> xPosPbo,
                     GPU::Buffer<uint32_t> xNegPbo, GPU::Buffer<uint32_t> yPosPbo, GPU::Buffer<uint32_t> yNegPbo,
                     GPU::Buffer<uint32_t> zPosPbo, GPU::Buffer<uint32_t> zNegPbo, const Matrix33<double>& perspective,
                     bool equiangular, GPU::Stream stream);

Status reprojectRectilinear(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                            unsigned width, unsigned height, const Matrix33<double>& perspective, GPU::Stream stream);
Status reprojectEquirectangular(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream);
Status reprojectFullFrameFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                 unsigned width, unsigned height, const Matrix33<double>& perspective,
                                 GPU::Stream stream);
Status reprojectCircularFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream);
Status reprojectStereographic(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                              unsigned width, unsigned height, const Matrix33<double>& perspective, GPU::Stream stream);
}  // namespace Core
}  // namespace VideoStitch
