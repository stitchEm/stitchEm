// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/render/geometry.hpp"

#include "backend/cuda/gpuKernelDef.h"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"
#include "../surface.hpp"
#include "cuda/util.hpp"

namespace VideoStitch {
namespace Render {

#include <backend/common/render/geometry.gpu>

template <>
Status drawLine(GPU::Surface& dst, int64_t width, int64_t height, float aX, float aY, float bX, float bY, float t,
                uint32_t color, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  lineSourceKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), (unsigned)width, (unsigned)height, aX,
                                                           aY, bX, bY, t, color);
  return CUDA_STATUS;
}
template <>
Status drawLine(GPU::Buffer<uint32_t>& dst, int64_t width, int64_t height, float aX, float aY, float bX, float bY,
                float t, uint32_t color, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  lineKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), (unsigned)width, (unsigned)height, aX, aY, bX, bY, t,
                                                     color);
  return CUDA_STATUS;
}

template <>
Status drawDisk(GPU::Surface& dst, int64_t width, int64_t height, float aX, float aY, float thickness, uint32_t color,
                GPU::Stream stream) {
  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocksPerGrid((unsigned)Cuda::ceilDiv(width, threadsPerBlock.x),
                     (unsigned)Cuda::ceilDiv(height, threadsPerBlock.y), 1);
  diskSourceKernel<<<blocksPerGrid, threadsPerBlock, 0, stream.get()>>>(
      dst.get().surface(), (unsigned)width, (unsigned)height, (float)aX, (float)aY, thickness, color);
  return CUDA_STATUS;
}
template <>
Status drawDisk(GPU::Buffer<uint32_t>& dst, int64_t width, int64_t height, float aX, float aY, float thickness,
                uint32_t color, GPU::Stream stream) {
  dim3 threadsPerBlock(16, 16, 1);
  dim3 blocksPerGrid((unsigned)Cuda::ceilDiv(width, threadsPerBlock.x),
                     (unsigned)Cuda::ceilDiv(height, threadsPerBlock.y), 1);
  diskKernel<<<blocksPerGrid, threadsPerBlock, 0, stream.get()>>>(dst.get(), (unsigned)width, (unsigned)height,
                                                                  (float)aX, (float)aY, thickness, color);
  return CUDA_STATUS;
}

#define CIRCLE_FN(fnName, kernelName)                                                                                  \
  template <>                                                                                                          \
  Status fnName(GPU::Surface& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius,  \
                float outerSqrRadius, uint32_t color, GPU::Stream stream) {                                            \
    dim3 dimBlock(16, 16, 1);                                                                                          \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);          \
    kernelName##Source<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), (unsigned)width, (unsigned)height, \
                                                               centerX, centerY, innerSqrRadius, outerSqrRadius,       \
                                                               color);                                                 \
    return CUDA_STATUS;                                                                                                \
  }                                                                                                                    \
  template <>                                                                                                          \
  Status fnName(GPU::Buffer<uint32_t>& dst, int64_t width, int64_t height, float centerX, float centerY,               \
                float innerSqrRadius, float outerSqrRadius, uint32_t color, GPU::Stream stream) {                      \
    dim3 dimBlock(16, 16, 1);                                                                                          \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);          \
    kernelName<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), (unsigned)width, (unsigned)height, centerX, centerY, \
                                                       innerSqrRadius, outerSqrRadius, color);                         \
    return CUDA_STATUS;                                                                                                \
  }

CIRCLE_FN(drawCircle, circleKernel)
CIRCLE_FN(drawCircleTop, circleTKernel)
CIRCLE_FN(drawCircleBottom, circleBKernel)
CIRCLE_FN(drawCircleTopRight, circleTRKernel)
CIRCLE_FN(drawCircleBottomRight, circleBRKernel)

}  // namespace Render
}  // namespace VideoStitch
