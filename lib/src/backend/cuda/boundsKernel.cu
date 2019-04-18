// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/boundsKernel.hpp"

#include "deviceBuffer.hpp"
#include "deviceStream.hpp"
#include <cuda/error.hpp>
#include <cuda/util.hpp>

#define REDUCE_THREADS_PER_BLOCK 512

namespace VideoStitch {
namespace Core {

namespace {

/**
 * This kernel computes the OR of all pixels in each row, and pouts the result in
 * colHasImage
 * FIXME do it with parallel reduction
 */
__global__ void vertOrKernel(const uint32_t* __restrict__ contrib, uint32_t* __restrict__ colHasImage,
                             unsigned panoWidth, unsigned panoHeight) {
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < panoWidth) {
    uint32_t accum = 0;
    for (unsigned row = 0; row < panoHeight; ++row) {
      accum |= contrib[panoWidth * row + col];
    }
    colHasImage[col] = accum;
  }
}

__global__ void horizOrKernel(const uint32_t* __restrict__ contrib, uint32_t* __restrict__ rowHasImage,
                              unsigned panoWidth, unsigned panoHeight) {
  unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t* rowp = contrib + panoWidth * row;

  if (row < panoHeight) {
    uint32_t accum = 0;
    for (unsigned col = 0; col < panoWidth; ++col) {
      accum |= rowp[col];
    }
    rowHasImage[row] = accum;
  }
}

}  // namespace

Status vertOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
              GPU::Buffer<uint32_t> colHasImage, GPU::Stream stream) {
  dim3 dimBlock(REDUCE_THREADS_PER_BLOCK, 1, 1);
  const unsigned numBlocks = (unsigned)Cuda::ceilDiv(croppedWidth, dimBlock.x);
  dim3 dimGrid(numBlocks, 1, 1);
  vertOrKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(contrib.get(), colHasImage.get(), (unsigned)croppedWidth,
                                                       (unsigned)croppedHeight);
  return CUDA_STATUS;
}

Status horizOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
               GPU::Buffer<uint32_t> rowHasImage, GPU::Stream stream) {
  dim3 dimBlock(REDUCE_THREADS_PER_BLOCK, 1, 1);
  const unsigned numBlocks = (unsigned)Cuda::ceilDiv(croppedHeight, dimBlock.x);
  dim3 dimGrid(numBlocks, 1, 1);
  horizOrKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(contrib.get(), rowHasImage.get(), (unsigned)croppedWidth,
                                                        (unsigned)croppedHeight);
  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
