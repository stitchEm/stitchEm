// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef _FILTERKERNEL_H_
#define _FILTERKERNEL_H_

namespace VideoStitch {
namespace Core {

/**
 * Filter each @stride-th column of the input buffer similarly to what the wavelet transform does
 * @height must be a divisible by 2.
 * The filter lets odd rows intact and smoothes out even rows.
 */
__global__ void filterKernel(unsigned char* data, unsigned width, unsigned height, unsigned hStride, unsigned vStride,
                             bool wrapAround) {
  unsigned columnId = blockIdx.x * blockDim.x + threadIdx.x;

  if (columnId < width) {
    unsigned char* colp = data + columnId * hStride;
    unsigned step = width * hStride * vStride;
    {
      uint32_t prev = colp[step];
      // boundary condition
      {
        uint32_t pprev;
        if (wrapAround) {
          pprev = colp[step * (height - 1)];
        } else {
          pprev = prev;
        }
        uint32_t v = colp[0];
        colp[0] = (2 * v + pprev + prev) >> 2;
      }
      __syncthreads();  // because of if
      for (unsigned row = 2; row < height; row += 2) {
        uint32_t next = colp[step * (row + 1)];
        uint32_t v = colp[step * row];
        colp[step * row] = (2 * v + next + prev) >> 2;
        prev = next;
      }
    }
  }
}
}  // namespace Core
}  // namespace VideoStitch
#endif
