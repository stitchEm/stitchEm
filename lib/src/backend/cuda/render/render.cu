// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/render/render.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "cuda/util.hpp"

namespace VideoStitch {
namespace Render {

namespace {
/**
 * Dummy fill kernel.
 */
__global__ void fillKernel(uint32_t* dst, size_t width, size_t height, uint32_t value) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    dst[y * width + x] = value;
  }
}
}  // namespace

Status fillBuffer(GPU::Buffer<uint32_t> dst, uint32_t value, size_t width, size_t height, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  fillKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().raw(), width, height, value);
  return CUDA_STATUS;
}
}  // namespace Render
}  // namespace VideoStitch
