// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/reduce.hpp"

#include "backend/common/imageOps.hpp"
#include "backend/cuda/deviceBuffer.hpp"

#include <cuda/error.hpp>
#include <cuda/util.hpp>

#include <cuda_runtime.h>
#include <cassert>
#include <stdint.h>
#include <stdio.h>

namespace VideoStitch {
namespace Image {

// See "Optimizing Parallel Reduction in CUDA" Mark Haris

template <uint32_t (*getValue)(uint32_t)>
__global__ void reduceKernel(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, unsigned size) {
  extern __shared__ uint32_t sdata[];
  const unsigned tid = threadIdx.x;
  unsigned i = blockIdx.x * (BLOCKSIZE * 2) + tid;
  const unsigned gridSize = BLOCKSIZE * 2 * gridDim.x;
  uint32_t startVal = 0;
  while (i < size) {
    startVal += getValue(src[i]);
    if (i + BLOCKSIZE < size) {
      startVal += getValue(src[i + BLOCKSIZE]);
    }
    i += gridSize;
  }
  sdata[tid] = startVal;
  __syncthreads();
  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    // No need to sync, only one warp. But fermi needs volatile !
    volatile uint32_t* localSdata = sdata;
    localSdata[tid] += localSdata[tid + 32];
    localSdata[tid] += localSdata[tid + 16];
    localSdata[tid] += localSdata[tid + 8];
    localSdata[tid] += localSdata[tid + 4];
    localSdata[tid] += localSdata[tid + 2];
    localSdata[tid] += localSdata[tid + 1];
  }
  if (tid == 0) {
    dst[blockIdx.x] = sdata[0];
  }
}

namespace {
inline __device__ uint32_t identity(uint32_t value) { return value; }

inline __device__ uint32_t isSolid(uint32_t value) { return RGBA::a(value) > 0 ? 1 : 0; }

inline __device__ uint32_t solidIdentityOrZero(uint32_t value) {
  return RGBA::a(value) > 0 ? (Image::RGBA::r(value) + Image::RGBA::g(value) + Image::RGBA::b(value)) / 3 : 0;
}
}  // namespace

template <uint32_t (*getValue)(uint32_t)>
Status reduce(const uint32_t* src, uint32_t* work, std::size_t size, uint32_t& result) {
  const dim3 dimBlock(BLOCKSIZE);
  // The first pass uses the actual getValue.
  if (size > 1) {
    const dim3 dimGrid((unsigned)Cuda::ceilDiv(size, 2 * BLOCKSIZE));
    reduceKernel<getValue><<<dimGrid, dimBlock, 4 * BLOCKSIZE>>>(work, src, (unsigned)size);
    size = dimGrid.x;
    src = work;
    work = work + size;
  }
  // Other passes simply sum.
  while (size > 1) {
    const dim3 dimGrid((unsigned)Cuda::ceilDiv(size, 2 * BLOCKSIZE));
    reduceKernel<identity><<<dimGrid, dimBlock, 4 * BLOCKSIZE>>>(work, src, (unsigned)size);
    size = dimGrid.x;
    src = work;
    work = work + size;
  }
  FAIL_RETURN(CUDA_STATUS);
  return CUDA_ERROR(cudaMemcpy(&result, src, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

Status reduceSum(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size, uint32_t& result) {
  return reduce<identity>(src.get(), work.get(), size, result);
}

Status reduceSumSolid(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size, uint32_t& result) {
  return reduce<solidIdentityOrZero>(src.get(), work.get(), size, result);
}

Status reduceCountSolid(GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t size,
                        uint32_t& result) {
  return reduce<isSolid>(src.get(), work.get(), size, result);
}

}  // namespace Image
}  // namespace VideoStitch
