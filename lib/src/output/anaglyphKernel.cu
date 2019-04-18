// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "anaglyph.hpp"

#include "backend/common/imageOps.hpp"

#include "cuda/util.hpp"
#include "cuda/error.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {

using namespace Image;

namespace Output {

// http://www.site.uottawa.ca/~edubois/anaglyph/LeastSquaresHowToPhotoshop.pdf

__global__ void anaglyphColorLeftKernel(uint32_t* dst, const uint32_t* src, const int64_t height, const int64_t width) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const uint32_t val = src[y * width + x];
    const int32_t r = clamp8(RGBA::r(val));
    const int32_t g = clamp8(RGBA::g(val));
    const int32_t b = clamp8(RGBA::b(val));
    const uint32_t orig = dst[y * width + x];
    const int32_t ar = (RGBA::r(orig) << 10) + 447 * r + 460 * g + 168 * b;
    const int32_t ag = (RGBA::g(orig) << 10) - 63 * r - 63 * g - 25 * b;
    const int32_t ab = (RGBA::b(orig) << 10) - 49 * r - 51 * g - 17 * b;
    dst[y * width + x] =
        RGBA::pack(clamp8(ar >> 10), clamp8(ag > 0 ? ag >> 10 : 0), clamp8(ab > 0 ? ab >> 10 : 0), 0xff);
  }
}

Status anaglyphColorLeft(uint32_t* dst, const uint32_t* src, const int64_t height, const int64_t width) {
  const dim3 dimBlock2D(16, 16, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  anaglyphColorLeftKernel<<<dimGrid2D, dimBlock2D, 0, 0>>>(dst, src, height, width);
  return CUDA_STATUS;
}

__global__ void anaglyphColorRightKernel(uint32_t* dst, const uint32_t* src, const int64_t height,
                                         const int64_t width) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const uint32_t val = src[y * width + x];
    const int32_t r = clamp8(RGBA::r(val));
    const int32_t g = clamp8(RGBA::g(val));
    const int32_t b = clamp8(RGBA::b(val));
    const uint32_t orig = dst[y * width + x];
    const int32_t ar = (RGBA::r(orig) << 10) - 11 * r - 33 * g - 7 * b;
    const int32_t ag = (RGBA::g(orig) << 10) + 386 * r + 779 * g + 9 * b;
    const int32_t ab = (RGBA::b(orig) << 10) - 27 * r - 95 * g + 1264 * b;
    dst[y * width + x] =
        RGBA::pack(clamp8(ar > 0 ? ar >> 10 : 0), clamp8(ag >> 10), clamp8(ab > 0 ? ab >> 10 : 0), 0xff);
  }
}

Status anaglyphColorRight(uint32_t* dst, const uint32_t* src, const int64_t height, const int64_t width) {
  const dim3 dimBlock2D(16, 16, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  anaglyphColorRightKernel<<<dimGrid2D, dimBlock2D, 0, 0>>>(dst, src, height, width);
  return CUDA_STATUS;
}
}  // namespace Output
}  // namespace VideoStitch
