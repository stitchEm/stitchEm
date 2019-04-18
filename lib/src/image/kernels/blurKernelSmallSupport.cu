// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef _BLURKERNEL_SMALL_SUPPORT_H_
#define _BLURKERNEL_SMALL_SUPPORT_H_

#include "backend/common/imageOps.hpp"

namespace VideoStitch {
namespace Image {
/**
 * Assumes 2 * radius < blockDim.x.
 */
#define gaussianBlur1DRGBA210SSKernelInterior POktmLly
template <uint32_t (*unrolledKernel)(const int32_t*)>
__global__ void gaussianBlur1DRGBA210SSKernelInterior(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src,
                                                      int w, int h, const int r) {
  // load to shared mem we must load radius more pixels on the left and right
  const int rowOffset = blockIdx.x * blockDim.x + threadIdx.x;
  // shMem holds unpacked ARGBARGBARGB...
  extern __shared__ int32_t shMem[];
  int32_t* col = shMem + 4 * threadIdx.x;
  int32_t v = 0;
  if (rowOffset < w) {
    v = src[blockIdx.y * w + rowOffset];
  }
  col[0] = RGB210::a(v);
  col[1] = RGB210::r(v);
  col[2] = RGB210::g(v);
  col[3] = RGB210::b(v);
  if (rowOffset + blockDim.x < w && threadIdx.x < 2 * r) {
    v = src[blockIdx.y * w + rowOffset + blockDim.x];
    col[blockDim.x] = RGB210::a(v);
    col[blockDim.x + 1] = RGB210::r(v);
    col[blockDim.x + 2] = RGB210::g(v);
    col[blockDim.x + 3] = RGB210::b(v);
  }
  __syncthreads();

  // Compute convolution.
  if (rowOffset + 2 * r < w) {
    dst[blockIdx.y * w + r + threadIdx.x] = unrolledKernel(col);
  }
}

#define gaussianBlur1DRGBA210SSKernelWrap gepMfedS
template <uint32_t (*unrolledKernel)(const int32_t*)>
__global__ void gaussianBlur1DRGBA210SSKernelWrap(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, int w,
                                                  int h, const int r) {
  // Load the r pixels before the right boundary
  // Load the r pixels of the right boundary
  // Load the r pixels of the left boundary
  // Load the r pixels after the left bounary
  // If r == 2 the pattern is:
  // src is read by threads:
  //   0 1 2 3 - - - - - - - 4 5 6 7
  // and written to shared mem as:
  //   4 5 6 7 0 1 2 3
  extern __shared__ int32_t shMem[];
  int32_t* col;
  int32_t v;
  if (threadIdx.x < 2 * r) {
    v = src[blockIdx.y * w + threadIdx.x];
    col = shMem + 4 * (2 * r + threadIdx.x);
  } else if (threadIdx.x < 4 * r) {
    v = src[(blockIdx.y + 1) * w - 4 * r + threadIdx.x];
    col = shMem + 4 * (threadIdx.x - 2 * r);
  }
  col[0] = RGB210::a(v);
  col[1] = RGB210::r(v);
  col[2] = RGB210::g(v);
  col[3] = RGB210::b(v);
  __syncthreads();
  // Now threads 4 and 5 will compute blur for pixels 6 and 7, and 6 and 7 for 0 and 1
  if (threadIdx.x >= 2 * r) {
    dst[blockIdx.y * w + threadIdx.x - 3 * r + (threadIdx.x < 3 * r) * w] = unrolledKernel(col);
  }
}

#define gaussianBlur1DRGBA210SSKernelNoWrap fTHeRdki
template <uint32_t (*unrolledKernel)(const int32_t*)>
__global__ void gaussianBlur1DRGBA210SSKernelNoWrap(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src, int w,
                                                    int h, const int r) {
  // Load the r pixels before the right boundary
  // Load the r pixels of the right boundary
  // Load the r pixels of the left boundary
  // Load the r pixels after the left bounary
  // If r == 2 the pattern is:
  // src is read by threads:
  //   0 1 2 3 - - - - - - - 4 5 6 7
  // and written to shared mem as:
  //   - - 0 1 2 3 4 5 6 7 - -
  //   0 0 0 1 2 3 4 5 6 7 7 7
  extern __shared__ int32_t shMem[];
  int32_t* col;
  int32_t v;
  if (threadIdx.x < 2 * r) {
    v = src[blockIdx.y * w + threadIdx.x];
    col = shMem + 4 * (2 * r + threadIdx.x);
  } else if (threadIdx.x < 4 * r) {
    v = src[(blockIdx.y + 1) * w - 4 * r + threadIdx.x];
    col = shMem + 4 * (threadIdx.x - 2 * r);
  }
  col[0] = RGB210::a(v);
  col[1] = RGB210::r(v);
  col[2] = RGB210::g(v);
  col[3] = RGB210::b(v);
  __syncthreads();
  // Now threads 4 and 5 will compute blur for pixels 6 and 7, and 6 and 7 for 0 and 1
  if (threadIdx.x >= 2 * r) {
    dst[blockIdx.y * w + threadIdx.x - 3 * r + (threadIdx.x < 3 * r) * w] = unrolledKernel(col);
  }
}
}  // namespace Image
}  // namespace VideoStitch
#endif
