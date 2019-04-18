// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/input/maskInput.hpp"

#include "../surface.hpp"
#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "cuda/util.hpp"

namespace VideoStitch {
namespace Input {

namespace {
__global__ void maskInputKernel(cudaSurfaceObject_t buffer, unsigned width, unsigned height,
                                const unsigned char* __restrict__ mask) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    if (mask[width * y + x] == 1) {
      surf2Dwrite(0, buffer, x * sizeof(uint32_t), y);
    }
  }
}
}  // namespace

Status maskInput(GPU::Surface& dst, GPU::Buffer<const unsigned char> maskDevBufferP, unsigned width, unsigned height,
                 GPU::Stream stream) {
  dim3 dimBlock(32, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  maskInputKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), width, height, maskDevBufferP.get());
  return CUDA_STATUS;
}

}  // namespace Input
}  // namespace VideoStitch

/// @endcond
