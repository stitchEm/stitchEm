// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "fill.hpp"

#include "../cuda/util.hpp"
#include "cuda/error.hpp"

#include <stdio.h>

namespace VideoStitch {
namespace Image {

namespace {
/**
 * A kernel that fills the buffer with a color.
 */
__global__ void fillKernel(uint32_t* dst, unsigned width, unsigned height, int32_t color) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    dst[y * width + x] = color;
  }
}
}  // namespace

Status fill(uint32_t* devBuffer, int64_t width, int64_t height, int32_t color, cudaStream_t stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv((unsigned)width, dimBlock.x),
               (unsigned)Cuda::ceilDiv((unsigned)height, dimBlock.y), 1);
  fillKernel<<<dimGrid, dimBlock, 0, stream>>>(devBuffer, (unsigned)width, (unsigned)height, color);
  return CUDA_STATUS;
}
}  // namespace Image
}  // namespace VideoStitch
