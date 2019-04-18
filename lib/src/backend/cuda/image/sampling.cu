// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/sampling.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"
#include "../surface.hpp"
#include "../gpuKernelDef.h"

#include "backend/common/vectorOps.hpp"
#include "cuda/util.hpp"
#include "image/kernels/sharedUtils.hpp"
#include "backend/cuda/core1/kernels/samplingKernel.cu"
#include <cuda_runtime.h>
#include <cassert>

#include "backend/common/image/sampling.gpu"

namespace VideoStitch {
namespace Image {

// ------------------- Subsampling

template <>
Status subsample22(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t srcWidth,
                   std::size_t srcHeight, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RegularKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)srcWidth,
                                                               (unsigned)srcHeight, (unsigned)dstWidth,
                                                               (unsigned)dstHeight);
  }
  // right boundary
  if (srcWidth & 1) {
    dim3 dimBlock(1, 256, 1);
    dim3 dimGrid(1, (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RightBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                     (unsigned)srcWidth, (unsigned)srcHeight,
                                                                     (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // bottom boundary
  if (srcHeight & 1) {
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), 1, 1);
    subsample22BottomBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                      (unsigned)srcWidth, (unsigned)srcHeight,
                                                                      (unsigned)dstWidth, (unsigned)dstHeight);
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy
    unsigned char* dstPtr = (unsigned char*)dst.get().raw();
    unsigned char* srcPtr = (unsigned char*)src.get().raw();
    return CUDA_ERROR(cudaMemcpyAsync(dstPtr + dstWidth * dstHeight - 1, srcPtr + srcHeight * srcWidth - 1,
                                      sizeof(unsigned char), cudaMemcpyDeviceToDevice, stream));
  }
  return CUDA_STATUS;
}

template <>
Status subsample22(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, std::size_t srcWidth, std::size_t srcHeight,
                   GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RegularKernelFloat2<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                     (unsigned)srcWidth, (unsigned)srcHeight,
                                                                     (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // right boundary
  if (srcWidth & 1) {
    dim3 dimBlock(1, 256, 1);
    dim3 dimGrid(1, (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RightBoundaryKernelFloat2<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                           (unsigned)srcWidth, (unsigned)srcHeight,
                                                                           (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // bottom boundary
  if (srcHeight & 1) {
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), 1, 1);
    subsample22BottomBoundaryKernelFloat2<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                            (unsigned)srcWidth, (unsigned)srcHeight,
                                                                            (unsigned)dstWidth, (unsigned)dstHeight);
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy
    float2* dstPtr = (float2*)dst.get().raw();
    float2* srcPtr = (float2*)src.get().raw();
    return CUDA_ERROR(cudaMemcpyAsync(dstPtr + dstWidth * dstHeight - 1, srcPtr + srcHeight * srcWidth - 1,
                                      sizeof(float2), cudaMemcpyDeviceToDevice, stream));
  }
  return CUDA_STATUS;
}

template <>
Status subsample22Mask(GPU::Buffer<float2> dst, GPU::Buffer<uint32_t> dstMask, GPU::Buffer<const float2> src,
                       GPU::Buffer<const uint32_t> srcMask, std::size_t srcWidth, std::size_t srcHeight,
                       unsigned blockSize, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22MaskRegularKernel<<<dimGrid, dimBlock, 0, stream>>>(
        dst.get().raw(), dstMask.get().raw(), src.get().raw(), srcMask.get().raw(), (unsigned)srcWidth,
        (unsigned)srcHeight, (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // right boundary
  if (srcWidth & 1) {
    dim3 dimBlock(1, blockSize * blockSize, 1);
    dim3 dimGrid(1, (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22MaskRightBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(
        dst.get().raw(), dstMask.get().raw(), src.get().raw(), srcMask.get().raw(), (unsigned)srcWidth,
        (unsigned)srcHeight, (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // bottom boundary
  if (srcHeight & 1) {
    dim3 dimBlock(blockSize * blockSize, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), 1, 1);
    subsample22MaskBottomBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(
        dst.get().raw(), dstMask.get().raw(), src.get().raw(), srcMask.get().raw(), (unsigned)srcWidth,
        (unsigned)srcHeight, (unsigned)dstWidth, (unsigned)dstHeight);
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy
    float2* dstPtr = (float2*)dst.get().raw();
    float2* srcPtr = (float2*)src.get().raw();
    FAIL_RETURN(CUDA_ERROR(cudaMemcpyAsync(dstPtr + dstWidth * dstHeight - 1, srcPtr + srcHeight * srcWidth - 1,
                                           sizeof(float2), cudaMemcpyDeviceToDevice, stream)));

    uint32_t* dstMaskPtr = (uint32_t*)dstMask.get().raw();
    uint32_t* srcMaskPtr = (uint32_t*)srcMask.get().raw();
    FAIL_RETURN(CUDA_ERROR(cudaMemcpyAsync(dstMaskPtr + dstWidth * dstHeight - 1, srcMaskPtr + srcHeight * srcWidth - 1,
                                           sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream)));
  }
  return CUDA_STATUS;
}

template <typename T>
Status subsample22Nearest(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t srcWidth, std::size_t srcHeight,
                          unsigned blockSize, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22NearestRegularKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                      (unsigned)srcWidth, (unsigned)srcHeight,
                                                                      (unsigned)dstWidth, (unsigned)dstHeight);
  }
  // right boundary
  if (srcWidth & 1) {
    dim3 dimBlock(1, blockSize * blockSize, 1);
    dim3 dimGrid(1, (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22NearestRightBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                            (unsigned)srcWidth, (unsigned)srcHeight,
                                                                            (unsigned)dstWidth, (unsigned)dstHeight);
  }
  if (srcHeight & 1) {
    dim3 dimBlock(blockSize * blockSize, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), 1, 1);
    subsample22NearestBottomBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(),
                                                                             (unsigned)srcWidth, (unsigned)srcHeight,
                                                                             (unsigned)dstWidth, (unsigned)dstHeight);
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy
    T* dstPtr = (T*)dst.get().raw();
    T* srcPtr = (T*)src.get().raw();
    return CUDA_ERROR(cudaMemcpyAsync(dstPtr + dstWidth * dstHeight - 1, srcPtr + srcHeight * srcWidth - 1, sizeof(T),
                                      cudaMemcpyDeviceToDevice, stream));
  }
  return CUDA_STATUS;
}

// template
// Status subsample22Nearest(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t srcWidth,
// std::size_t srcHeight, unsigned blockSize, GPU::Stream stream); template Status
// subsample22Nearest(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t srcWidth,
// std::size_t srcHeight, unsigned blockSize, GPU::Stream stream); template Status subsample22Nearest(GPU::Buffer<float>
// dst, GPU::Buffer<const float> src, std::size_t srcWidth, std::size_t srcHeight, unsigned blockSize, GPU::Stream
// stream);

Status subsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t srcWidth,
                       std::size_t srcHeight, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RGBARegularKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), src.get(), (unsigned)srcWidth,
                                                                   (unsigned)srcHeight, (unsigned)dstWidth);
  }
  // right boundary
  if (srcWidth & 1) {
    dim3 dimBlock(1, 256, 1);
    dim3 dimGrid(1, (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
    subsample22RGBARightBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), src.get(), (unsigned)srcWidth,
                                                                         (unsigned)srcHeight, (unsigned)dstWidth);
  }
  if (srcHeight & 1) {
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), 1, 1);
    subsample22RGBABottomBoundaryKernel<<<dimGrid, dimBlock, 0, stream>>>(dst.get(), src.get(), (unsigned)srcWidth,
                                                                          (unsigned)srcHeight, (unsigned)dstWidth);
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy
    uint32_t* dstPtr = (uint32_t*)dst.get().raw();
    uint32_t* srcPtr = (uint32_t*)src.get().raw();
    return CUDA_ERROR(cudaMemcpyAsync(dstPtr + dstWidth * dstHeight - 1, srcPtr + srcHeight * srcWidth - 1, 4,
                                      cudaMemcpyDeviceToDevice, stream));
  }
  return CUDA_STATUS;
}

// ------------------ Upsampling

/**
 * Upsample @src by a factor of two on each dimension and put it in into @dst.
 * @dst has size (@dstWidth * @dstHeight), @dst has size ((@dstWidth + 1)/2 * (@dstHeight + 1)/2).
 * This is more complex than subsampling since we need to interpolate at the same time.
 * We use shared memory to share reads to global memory between threads.
 * In addition, we make sure that memory accesses are coalesced.
 * To avoid divergence in the regular case, there are two kernels: one that applies inside
 * the image, and one that applies to boundaries.
 * The alpha is taken to be solid if at least one sample is solid.
 */

/**
  // Bilinear interpolation.
  //                           +=======+=======+=======+
  //                           |       |       |       |
  //                           |   A   |   B   |   C   |
  //  |       |       |        |       |       |       |
  //  +=======+=======+=       +=======+===+===+=======+
  //  |       |       |        |       | a | b |       |
  //  |   D   |   E   |        |   D   +---+---+   F   |
  //  |       |       |        |       | c | d |       |
  //  +=======+=======+=  =>   +=======+===+===+=======+
  //  |       |       |        |       |       |       |
  //  |   G   |   H   |        |   G   |   H   |   I   |
  //  |       |       |        |       |       |       |
  //  +=======+=======+=       +=======+=======+=======+
  //
  // The current thread loads source pixel E, then computes interpolated values for a, b, c, d:
  //    a = 1 / 16 * A + 3 / 16 * [D + B] + 9 / 16 * E
  //    b = 1 / 16 * C + 3 / 16 * [B + F] + 9 / 16 * E
  //    c = 1 / 16 * G + 3 / 16 * [D + H] + 9 / 16 * E
  //    d = 1 / 16 * I + 3 / 16 * [F + H] + 9 / 16 * E
*/

struct BilinearInterpolationRGB210 {
  typedef uint32_t Type;

  static inline __device__ uint32_t interpolate(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    // see above
    const int32_t alphaA = RGB210::a(a);
    const int32_t alphaB = RGB210::a(b);
    const int32_t alphaC = RGB210::a(c);
    const int32_t alphaD = RGB210::a(d);
    const int32_t divisor = 9 * alphaA + 3 * (alphaB + alphaC) + alphaD;
    return RGB210::pack(
        (alphaA * 9 * RGB210::r(a) + 3 * (alphaB * RGB210::r(b) + alphaC * RGB210::r(c)) + alphaD * RGB210::r(d)) /
            divisor,
        (alphaA * 9 * RGB210::g(a) + 3 * (alphaB * RGB210::g(b) + alphaC * RGB210::g(c)) + alphaD * RGB210::g(d)) /
            divisor,
        (alphaA * 9 * RGB210::b(a) + 3 * (alphaB * RGB210::b(b) + alphaC * RGB210::b(c)) + alphaD * RGB210::b(d)) /
            divisor,
        divisor > 0);
  }
};

struct BilinearInterpolationRGBA {
  typedef uint32_t Type;

  static inline __device__ uint32_t interpolate(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    // see above
    const uint32_t alphaA = !!RGBA::a(a);
    const uint32_t alphaB = !!RGBA::a(b);
    const uint32_t alphaC = !!RGBA::a(c);
    const uint32_t alphaD = !!RGBA::a(d);
    const uint32_t divisor = 9 * alphaA + 3 * (alphaB + alphaC) + alphaD;
    if (divisor) {
      return RGBASolid::pack(
          (alphaA * 9 * RGBA::r(a) + 3 * (alphaB * RGBA::r(b) + alphaC * RGBA::r(c)) + alphaD * RGBA::r(d)) / divisor,
          (alphaA * 9 * RGBA::g(a) + 3 * (alphaB * RGBA::g(b) + alphaC * RGBA::g(c)) + alphaD * RGBA::g(d)) / divisor,
          (alphaA * 9 * RGBA::b(a) + 3 * (alphaB * RGBA::b(b) + alphaC * RGBA::b(c)) + alphaD * RGBA::b(d)) / divisor,
          0xff);
    } else {
      return 0;
    }
  }
};

template <typename T>
struct BilinearInterpolation {
  typedef T Type;

  static inline __device__ T interpolate(T a, T b, T c, T d) {
    // see above
    return (T)(9.0f / 16.0f * a + 3.0f / 16.0f * (b + c) + 1.0f / 16.0f * d);
  }
};

Status upsample22RGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                         std::size_t dstHeight, bool wrap, GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(srcWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(srcHeight, dimBlock.y), 1);
  if (wrap) {
    upsample22Kernel<HWrapBoundary<uint32_t>, BilinearInterpolationRGB210>
        <<<dimGrid, dimBlock, (16 + 2) * (16 + 2) * 4, stream.get()>>>(dst.get(), src.get(), (unsigned)dstWidth,
                                                                       (unsigned)dstHeight, srcWidth, srcHeight);
  } else {
    upsample22Kernel<ExtendBoundary<uint32_t>, BilinearInterpolationRGB210>
        <<<dimGrid, dimBlock, (16 + 2) * (16 + 2) * 4, stream.get()>>>(dst.get(), src.get(), (unsigned)dstWidth,
                                                                       (unsigned)dstHeight, srcWidth, srcHeight);
  }
  return CUDA_STATUS;
}

Status upsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                      std::size_t dstHeight, bool wrap, GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(srcWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(srcHeight, dimBlock.y), 1);
  if (wrap) {
    upsample22Kernel<HWrapBoundary<uint32_t>, BilinearInterpolationRGBA>
        <<<dimGrid, dimBlock, (16 + 2) * (16 + 2) * 4, stream.get()>>>(dst.get(), src.get(), (unsigned)dstWidth,
                                                                       (unsigned)dstHeight, srcWidth, srcHeight);
  } else {
    upsample22Kernel<ExtendBoundary<uint32_t>, BilinearInterpolationRGBA>
        <<<dimGrid, dimBlock, (16 + 2) * (16 + 2) * 4, stream.get()>>>(dst.get(), src.get(), (unsigned)dstWidth,
                                                                       (unsigned)dstHeight, srcWidth, srcHeight);
  }
  return CUDA_STATUS;
}

template <typename T>
Status upsample22(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t dstWidth, std::size_t dstHeight, bool wrap,
                  GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(srcWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(srcHeight, dimBlock.y), 1);
  if (wrap) {
    upsample22Kernel<HWrapBoundary<T>, BilinearInterpolation<T>><<<dimGrid, dimBlock, 0, stream.get()>>>(
        dst.get(), src.get(), (unsigned)dstWidth, (unsigned)dstHeight, srcWidth, srcHeight);
  } else {
    upsample22Kernel<ExtendBoundary<T>, BilinearInterpolation<T>><<<dimGrid, dimBlock, 0, stream.get()>>>(
        dst.get(), src.get(), (unsigned)dstWidth, (unsigned)dstHeight, srcWidth, srcHeight);
  }
  return CUDA_STATUS;
}

template Status upsample22(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                           std::size_t dstHeight, bool wrap, GPU::Stream stream);
template Status upsample22(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t dstWidth,
                           std::size_t dstHeight, bool wrap, GPU::Stream stream);
template Status upsample22(GPU::Buffer<float> dst, GPU::Buffer<const float> src, std::size_t dstWidth,
                           std::size_t dstHeight, bool wrap, GPU::Stream stream);
template Status upsample22(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, std::size_t dstWidth,
                           std::size_t dstHeight, bool wrap, GPU::Stream stream);

// ---------------- Masks sampling

Status subsampleMask22(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t srcWidth,
                       std::size_t srcHeight, unsigned blockSize, GPU::Stream stream) {
  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
  subsampleMask22Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)srcWidth,
                                                                (unsigned)srcHeight, (unsigned)dstWidth);
  return CUDA_STATUS;
}

}  // namespace Image
}  // namespace VideoStitch
