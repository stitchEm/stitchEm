// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cuda_runtime.h>
#include "../cuda/error.hpp"
#include "../cuda/util.hpp"

#include <backend/cuda/deviceStream.hpp>

#include "transpose.hpp"
#include <stdint.h>

#define CUDABLOCKSIZE 16

namespace VideoStitch {
namespace Image {

/**
 * Transpose the general (not necessarily square) matrix @src
 * width @srcWidth columns and @srcHeight rows into dst.
 * @srcWidth and @srcHeight must be multiples of the tile size.
 */
template <unsigned CudaBlockSize, typename T>
__global__ void transposeKernel(T *dst, const T *src, unsigned srcWidth, unsigned srcHeight) {
  __shared__ T tile[CudaBlockSize][CudaBlockSize + 1];

  unsigned xIndex = blockIdx.x * CudaBlockSize + threadIdx.x;
  unsigned yIndex = blockIdx.y * CudaBlockSize + threadIdx.y;
  unsigned tileInStart = xIndex + yIndex * srcWidth;

  xIndex = blockIdx.y * CudaBlockSize + threadIdx.x;
  yIndex = blockIdx.x * CudaBlockSize + threadIdx.y;
  unsigned tileOutStart = xIndex + yIndex * srcHeight;

  for (int i = 0; i < CudaBlockSize; i += CudaBlockSize) {
    tile[threadIdx.y + i][threadIdx.x] = src[tileInStart + i * srcWidth];
  }

  __syncthreads();

  for (int i = 0; i < CudaBlockSize; i += CudaBlockSize) {
    dst[tileOutStart + i * srcHeight] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/**
 * Transpose the general (not necessarily square) matrix @src
 * width @srcWidth columns and @srcHeight rows into dst.
 * @srcWidth and @srcHeight are arbitrary.
 */
template <unsigned CudaBlockSize, typename T>
__global__ void transposeGenericKernel(T *dst, const T *src, unsigned srcWidth, unsigned srcHeight) {
  __shared__ T tile[CudaBlockSize][CudaBlockSize + 1];

  const unsigned xSrcIndex = blockIdx.x * CudaBlockSize + threadIdx.x;
  const unsigned ySrcIndex = blockIdx.y * CudaBlockSize + threadIdx.y;
  const unsigned tileInStart = xSrcIndex + ySrcIndex * srcWidth;

  const unsigned xDstIndex = blockIdx.y * CudaBlockSize + threadIdx.x;
  const unsigned yDstIndex = blockIdx.x * CudaBlockSize + threadIdx.y;
  const unsigned tileOutStart = xDstIndex + yDstIndex * srcHeight;

  if (xSrcIndex < srcWidth && ySrcIndex < srcHeight) {
    tile[threadIdx.y][threadIdx.x] = src[tileInStart];
  }

  __syncthreads();
  if (xDstIndex < srcHeight && yDstIndex < srcWidth) {
    dst[tileOutStart] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
Status transpose(T *dst, const T *src, int64_t w, int64_t h, GPU::Stream &stream) {
  dim3 dimBlock(CUDABLOCKSIZE, CUDABLOCKSIZE);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(w, CUDABLOCKSIZE), (unsigned)Cuda::ceilDiv(h, CUDABLOCKSIZE));
  transposeGenericKernel<CUDABLOCKSIZE, T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst, src, (unsigned)w, (unsigned)h);
  return CUDA_STATUS;
}

// explicit template instanciations
template Status transpose(uint32_t *dst, const uint32_t *src, int64_t w, int64_t h, GPU::Stream &stream);
template Status transpose(unsigned char *dst, const unsigned char *src, int64_t w, int64_t h, GPU::Stream &stream);
template Status transpose(float *dst, const float *src, int64_t w, int64_t h, GPU::Stream &stream);
template Status transpose(float2 *dst, const float2 *src, int64_t w, int64_t h, GPU::Stream &stream);
}  // namespace Image
}  // namespace VideoStitch
