// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imgInsert.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"

#include "cuda/util.hpp"

#include <cuda_runtime.h>
#include <cassert>

const unsigned int CudaBlockSize = 16;

namespace VideoStitch {
namespace Image {

/**
 * This kernel inserts the content of the (packed) image @src at offset (@offsetX, offsetY) into @dest.
 * On overflow, the image wraps if hWrap (resp vWrap) is true. Else nothing is written.
 * Pixels with zero alpha are not merged.
 * 2D version: We assume that the @src (but not the dst) image is divisible
 * by the block size on each dimension.
 */
#define DEFINE_IMGIIK(funcName, testPredicate, dstIndexComputation)                                                   \
  __global__ void funcName(uint32_t* __restrict__ dst, unsigned dstWidth, unsigned dstHeight,                         \
                           const uint32_t* __restrict__ src, unsigned srcWidth, unsigned srcHeight, unsigned offsetX, \
                           unsigned offsetY) {                                                                        \
    unsigned srcX = blockIdx.x * blockDim.x + threadIdx.x;                                                            \
    unsigned srcY = blockIdx.y * blockDim.y + threadIdx.y;                                                            \
                                                                                                                      \
    unsigned dstX = srcX + offsetX;                                                                                   \
    unsigned dstY = srcY + offsetY;                                                                                   \
                                                                                                                      \
    if (srcX < srcWidth && srcY < srcHeight) {                                                                        \
      uint32_t p = src[srcWidth * srcY + srcX];                                                                       \
      if (testPredicate) {                                                                                            \
        dst[dstIndexComputation] = p;                                                                                 \
      }                                                                                                               \
    }                                                                                                                 \
  }

/**
 * Same as above, except that @mask is used for blending.
 */
#define DEFINE_IMGIIKM(funcName, testPredicate, dstIndexComputation)                                                  \
  template <typename PixelType>                                                                                       \
  __global__ void funcName(uint32_t* __restrict__ dst, unsigned dstWidth, unsigned dstHeight,                         \
                           const uint32_t* __restrict__ src, unsigned srcWidth, unsigned srcHeight, unsigned offsetX, \
                           unsigned offsetY, const unsigned char* __restrict__ mask) {                                \
    unsigned srcX = blockIdx.x * blockDim.x + threadIdx.x;                                                            \
    unsigned srcY = blockIdx.y * blockDim.y + threadIdx.y;                                                            \
                                                                                                                      \
    unsigned dstX = srcX + offsetX;                                                                                   \
    unsigned dstY = srcY + offsetY;                                                                                   \
                                                                                                                      \
    if (srcX < srcWidth && srcY < srcHeight) {                                                                        \
      unsigned srcIndex = srcWidth * srcY + srcX;                                                                     \
      uint32_t p = src[srcIndex];                                                                                     \
      if (testPredicate) {                                                                                            \
        unsigned dstIndex = dstIndexComputation;                                                                      \
        uint32_t q = dst[dstIndex];                                                                                   \
        if (PixelType::a(q)) {                                                                                        \
          int32_t m = mask[srcIndex];                                                                                 \
          uint32_t mR = (m * PixelType::r(p) + (255 - m) * PixelType::r(q)) / 255;                                    \
          uint32_t mG = (m * PixelType::g(p) + (255 - m) * PixelType::g(q)) / 255;                                    \
          uint32_t mB = (m * PixelType::b(p) + (255 - m) * PixelType::b(q)) / 255;                                    \
          q = PixelType::pack(mR, mG, mB, 0xff);                                                                      \
        } else {                                                                                                      \
          if (mask[srcIndex]) {                                                                                       \
            q = p;                                                                                                    \
          }                                                                                                           \
        }                                                                                                             \
        dst[dstIndex] = q;                                                                                            \
      }                                                                                                               \
    }                                                                                                                 \
  }

DEFINE_IMGIIKM(imgInsertIntoKernelMaskedNoWrap, RGB210::a(p) && dstX < dstWidth && dstY < dstHeight,
               dstWidth* dstY + dstX)
DEFINE_IMGIIK(imgInsertIntoKernelNoWrap, RGB210::a(p) && dstX < dstWidth && dstY < dstHeight, dstWidth* dstY + dstX)
DEFINE_IMGIIKM(imgInsertIntoKernelMaskedHWrap, RGB210::a(p) && dstY < dstHeight, dstWidth* dstY + (dstX % dstWidth))
DEFINE_IMGIIK(imgInsertIntoKernelHWrap, RGB210::a(p) && dstY < dstHeight, dstWidth* dstY + (dstX % dstWidth))
DEFINE_IMGIIKM(imgInsertIntoKernelMaskedVWrap, RGB210::a(p) && dstX < dstWidth, dstWidth*(dstY % dstHeight) + dstX)
DEFINE_IMGIIK(imgInsertIntoKernelVWrap, RGB210::a(p) && dstX < dstWidth, dstWidth*(dstY % dstHeight) + dstX)

template <typename PixelType>
Status imgInsertInto(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                     GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                     std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap, bool vWrap,
                     GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(srcWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(srcHeight, dimBlock.y), 1);
  if (mask.wasAllocated()) {
    if (hWrap) {
      if (vWrap) {
        assert(false);
      } else {
        imgInsertIntoKernelMaskedHWrap<PixelType><<<dimGrid, dimBlock, 0, stream>>>(
            dst.get(), (unsigned)dstWidth, (unsigned)dstHeight, src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
            (unsigned)offsetX, (unsigned)offsetY, mask.get());
      }
    } else {
      if (vWrap) {
        imgInsertIntoKernelMaskedVWrap<PixelType><<<dimGrid, dimBlock, 0, stream>>>(
            dst.get(), (unsigned)dstWidth, (unsigned)dstHeight, src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
            (unsigned)offsetX, (unsigned)offsetY, mask.get());
      } else {
        imgInsertIntoKernelMaskedNoWrap<PixelType><<<dimGrid, dimBlock, 0, stream>>>(
            dst.get(), (unsigned)dstWidth, (unsigned)dstHeight, src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
            (unsigned)offsetX, (unsigned)offsetY, mask.get());
      }
    }
  } else {
    if (hWrap) {
      if (vWrap) {
        assert(false);
      } else {
        imgInsertIntoKernelHWrap<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), (unsigned)dstWidth, (unsigned)dstHeight,
                                                                   src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
                                                                   (unsigned)offsetX, (unsigned)offsetY);
      }
    } else {
      if (vWrap) {
        imgInsertIntoKernelVWrap<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), (unsigned)dstWidth, (unsigned)dstHeight,
                                                                   src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
                                                                   (unsigned)offsetX, (unsigned)offsetY);
      } else {
        imgInsertIntoKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), (unsigned)dstWidth, (unsigned)dstHeight,
                                                                    src.get(), (unsigned)srcWidth, (unsigned)srcHeight,
                                                                    (unsigned)offsetX, (unsigned)offsetY);
      }
    }
  }
  return CUDA_STATUS;
}

Status imgInsertInto(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                     GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                     std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap, bool vWrap,
                     GPU::Stream gpuStream) {
  return imgInsertInto<Image::RGBA>(dst, dstWidth, dstHeight, src, srcWidth, srcHeight, offsetX, offsetY, mask, hWrap,
                                    vWrap, gpuStream);
}

Status imgInsertInto10bit(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                          GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight,
                          std::size_t offsetX, std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap,
                          bool vWrap, GPU::Stream gpuStream) {
  return imgInsertInto<Image::RGB210>(dst, dstWidth, dstHeight, src, srcWidth, srcHeight, offsetX, offsetY, mask, hWrap,
                                      vWrap, gpuStream);
}

}  // namespace Image
}  // namespace VideoStitch
