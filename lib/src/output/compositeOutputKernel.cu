// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "compositeOutput.hpp"

#include "cuda/util.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {
namespace Output {

template <int pixelSize>
__global__ void writeHalfBufferHorizontalInterKernel(char* dst, const char* src, const int64_t height,
                                                     const int64_t dstWidth, const int64_t srcWidth,
                                                     const int64_t xOffset) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < srcWidth && y < height) {
#pragma unroll
    for (int k = 0; k < pixelSize; ++k) {
      dst[pixelSize * (y * dstWidth + x + xOffset) + k] = src[pixelSize * (y * srcWidth + x) + k];
    }
  }
}

template <int pixelSize>
void writeHalfBufferHorizontalInter(char* dst, const char* src, const int64_t height, const int64_t dstWidth,
                                    const int64_t srcWidth, const int64_t xOffset) {
  const dim3 dimBlock2D(16, 16, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(srcWidth, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y),
                       1);
  writeHalfBufferHorizontalInterKernel<pixelSize>
      <<<dimGrid2D, dimBlock2D, 0, 0>>>(dst, src, height, dstWidth, srcWidth, xOffset);
}

template void writeHalfBufferHorizontalInter<1>(char* dst, const char* src, const int64_t height,
                                                const int64_t dstWidth, const int64_t srcWidth, const int64_t xOffset);
template void writeHalfBufferHorizontalInter<2>(char* dst, const char* src, const int64_t height,
                                                const int64_t dstWidth, const int64_t srcWidth, const int64_t xOffset);
template void writeHalfBufferHorizontalInter<3>(char* dst, const char* src, const int64_t height,
                                                const int64_t dstWidth, const int64_t srcWidth, const int64_t xOffset);
template void writeHalfBufferHorizontalInter<4>(char* dst, const char* src, const int64_t height,
                                                const int64_t dstWidth, const int64_t srcWidth, const int64_t xOffset);

}  // namespace Output
}  // namespace VideoStitch
