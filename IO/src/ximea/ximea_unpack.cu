// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ximea_unpack.hpp"

inline int64_t ceilDiv(int64_t v, unsigned int d) {
  const int64_t res = v / (int64_t)d;
  return res + (int64_t)(v - res * (int64_t)d > 0);  // add one is the remainder is nonzero
}

/**
 * This kernel converts the buffer from Mono12p to 8-bits pixels.
 * Each thread manages 2 pixels, reading 3 bytes and writing 2.
 */
__global__ void unpackMono12pKernel(unsigned char* __restrict__ dst, const unsigned char* __restrict__ src,
                                    unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    unsigned s = 3 * i;
    unsigned d = 2 * i;
    dst[d] = (src[s] >> 4) | (src[s + 1] << 4);
    dst[d + 1] = src[s + 2];
  }
}

void unpackMono12p(unsigned char* dst, const unsigned char* src, int64_t width, int64_t height, cudaStream_t s) {
  const dim3 dimBlock2D(16, 16, 1);
  const dim3 dimGrid2D((unsigned)ceilDiv(width, dimBlock2D.x), (unsigned)ceilDiv(height / 2, dimBlock2D.y), 1);
  unpackMono12pKernel<<<dimGrid2D, dimBlock2D, 0, s>>>(dst, src, (unsigned)width, (unsigned)height / 2);
}
