// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imgExtract.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "cuda/util.hpp"

#include <cuda_runtime.h>
#include <cassert>

namespace VideoStitch {
namespace Image {

/**
 * This kernel extract a part of the content of the image @src at
 * offset (@offsetX, offsetY) with size (dstWidth x dstHeight) and writes it in into a packed buffer @dst.
 * @dst must be large enough to hold the dstWidth * dstHeight pixels.
 * On overflow, the source image wraps if hWrap is true. Else pixels are filled with 0.
 * 2D version: We assume that the @dst (but not the @src) image is divisible
 * by the block size on each dimension.
 */
__global__ void imgExtractFromKernel(uint32_t* __restrict__ dst, unsigned dstWidth, unsigned dstHeight,
                                     const uint32_t* __restrict__ src, unsigned srcWidth, unsigned srcHeight,
                                     int offsetX, int offsetY, bool hWrap) {
  int dstX = blockIdx.x * blockDim.x + threadIdx.x;
  int dstY = blockIdx.y * blockDim.y + threadIdx.y;

  int srcX = offsetX + dstX;
  int srcY = offsetY + dstY;

  uint32_t res = 0;
  if (dstX < dstWidth && dstY < dstHeight) {
    if (0 <= srcY && srcY < srcHeight) {
      if (hWrap) {
        if (0 <= srcX) {
          if (srcX < srcWidth) {
            res = src[srcWidth * srcY + srcX];
          } else {
            res = src[srcWidth * srcY + (srcX % srcWidth)];
          }
        } else {
          res = src[srcWidth * srcY + srcWidth + (srcX % srcWidth)];  // modulo has sign of dividend
        }
      } else if (0 <= srcX & srcX < srcWidth) {
        res = src[srcWidth * srcY + srcX];
      }
    }
    dst[dstWidth * dstY + dstX] = res;
  }
}

Status imgExtractFrom(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                      GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                      std::size_t offsetY, bool hWrap, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
  imgExtractFromKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), (unsigned)dstWidth, (unsigned)dstHeight,
                                                               src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
                                                               (unsigned)offsetX, (unsigned)offsetY,
                                                               hWrap);  // wraps
  return CUDA_STATUS;
}
}  // namespace Image
}  // namespace VideoStitch
