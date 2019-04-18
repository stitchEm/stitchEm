// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "fillRenderer.hpp"

#include "../cuda/util.hpp"
#include "cuda/error.hpp"

namespace VideoStitch {
namespace Render {

namespace {
/**
 * Fill rectangle kernel.
 */
__global__ void fillRectKernel(uint32_t* dst, uint32_t value, unsigned left, unsigned top, unsigned right,
                               unsigned bottom, unsigned bufferWidth) {
  const unsigned x = left + blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = top + blockIdx.y * blockDim.y + threadIdx.y;
  if (x < right && y < bottom) {
    dst[y * bufferWidth + x] = value;
  }
}

/**
 * Clamps a value between 0 and a maximum value.
 * @param value input value
 * @param maxValue max value
 */
int64_t clampZeroMax(int64_t value, int64_t maxValue) {
  if (value < 0) {
    return 0;
  } else if (value > maxValue) {
    return maxValue;
  }
  return value;
}
}  // namespace

Status FillRenderer::draw(uint32_t* dst, int64_t dstWidth, int64_t dstHeight, int64_t left, int64_t top, int64_t right,
                          int64_t bottom, uint32_t color, uint32_t /*bgcolor*/, cudaStream_t stream) const {
  left = clampZeroMax(left, dstWidth);
  right = clampZeroMax(right, dstWidth);
  top = clampZeroMax(top, dstHeight);
  bottom = clampZeroMax(bottom, dstHeight);
  int64_t width = right - left;
  int64_t height = bottom - top;
  if (width <= 0 || height <= 0) {
    return {Origin::GPU, ErrType::ImplementationError, "Negative size for rectangle filling"};
  }
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y));
  fillRectKernel<<<dimGrid, dimBlock, 0, stream>>>(dst, color, (unsigned)left, (unsigned)top, (unsigned)right,
                                                   (unsigned)bottom, (unsigned)dstWidth);
  return CUDA_STATUS;
}

}  // namespace Render
}  // namespace VideoStitch
