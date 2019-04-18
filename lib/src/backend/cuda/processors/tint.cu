// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/tint.hpp"

#include "backend/common/imageOps.hpp"
#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"

namespace VideoStitch {
namespace Core {

namespace {
/**
 * A kernel that tints everything with a color.
 */
__global__ void tintKernel(cudaSurfaceObject_t dst, unsigned width, unsigned height, int32_t r, int32_t g, int32_t b) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uint32_t srcColor;
    surf2Dread(&srcColor, dst, x * sizeof(uint32_t), y);
    // 0.2126 R + 0.7152 G + 0.0722 B
    const int32_t luminosity =
        1742 * Image::RGBA::r(srcColor) + 5859 * Image::RGBA::g(srcColor) + 591 * Image::RGBA::b(srcColor);
    uint32_t dstColor = Image::RGBA::pack((r * luminosity) >> 21, (g * luminosity) >> 21, (b * luminosity) >> 21,
                                          Image::RGBA::a(srcColor));
    surf2Dwrite(dstColor, dst, x * sizeof(uint32_t), y);
  }
}
}  // namespace

Status tint(GPU::Surface& dst, unsigned width, unsigned height, int32_t r, int32_t g, int32_t b, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  tintKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), width, height, r, g, b);
  return CUDA_STATUS;
}
}  // namespace Core
}  // namespace VideoStitch
