// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/maskoverlay.hpp"

#include "backend/common/imageOps.hpp"

#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"

namespace VideoStitch {
namespace Core {

namespace {
/**
 * A kernel that overlays the mask over the image.
 */
__global__ void maskOverlayKernel(cudaSurfaceObject_t dst, unsigned width, unsigned height, int32_t r, int32_t g,
                                  int32_t b, int32_t alpha) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uint32_t srcColor;
    surf2Dread(&srcColor, dst, x * sizeof(uint32_t), y);

    float overlayAlpha = float(alpha) / 255.f;

    if (VideoStitch::Image::RGBA::a(srcColor) < 255) {
      const int32_t overlayR = VideoStitch::Image::RGBA::r(srcColor) * (1 - overlayAlpha) + overlayAlpha * r;
      const int32_t overlayG = VideoStitch::Image::RGBA::g(srcColor) * (1 - overlayAlpha) + overlayAlpha * g;
      const int32_t overlayB = VideoStitch::Image::RGBA::b(srcColor) * (1 - overlayAlpha) + overlayAlpha * b;

      surf2Dwrite(VideoStitch::Image::RGBA::pack(overlayR, overlayG, overlayB, 255), dst, x * sizeof(uint32_t), y);
    }
  }
}
}  // namespace

Status maskOverlay(GPU::Surface& dst, unsigned width, unsigned height, uint32_t color, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  maskOverlayKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      dst.get().surface(), width, height, VideoStitch::Image::RGBA::r(color), VideoStitch::Image::RGBA::g(color),
      VideoStitch::Image::RGBA::b(color), VideoStitch::Image::RGBA::a(color));
  return CUDA_STATUS;
}
}  // namespace Core
}  // namespace VideoStitch
