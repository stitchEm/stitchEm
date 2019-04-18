// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/grid.hpp"

#include "cuda/util.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"

namespace VideoStitch {
namespace Core {

namespace {

#include "../gpuKernelDef.h"

#include <backend/common/input/grid.gpu>

}  // namespace

Status grid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth, uint32_t color,
            uint32_t bgColor, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  gridKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), width, height, size, lineWidth, color, bgColor);
  return CUDA_STATUS;
}

Status transparentForegroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t bgColor, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  transparentFGGridKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), width, height, size, lineWidth, bgColor);
  return CUDA_STATUS;
}

Status transparentBackgroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t color, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  transparentBGGridKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), width, height, size, lineWidth, color);
  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
