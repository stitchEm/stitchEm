// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/input/checkerBoard.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"

#include "cuda/util.hpp"

namespace VideoStitch {
namespace Input {

namespace {

#include "../gpuKernelDef.h"

#include <backend/common/input/checkerBoard.gpu>

}  // namespace

Status overlayCheckerBoard(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, unsigned checkerSize,
                           uint32_t color1, uint32_t color2, uint32_t color3, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  checkerBoardKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), width, height, checkerSize, color1, color2,
                                                             color3);
  return CUDA_STATUS;
}

}  // namespace Input
}  // namespace VideoStitch
