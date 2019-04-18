// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/blur.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "cuda/util.hpp"
#include "image/transpose.hpp"

#include "libvideostitch/profile.hpp"

#include <cuda_runtime.h>
#include <cassert>

#define RGBA_BOX_BLUR_1D_BLOCK_SIZE (4 * 32)
#define RGBA_BOX_BLUR_SS_1D_BLOCK_SIZE (4 * 32)

template <typename Type>
struct ScalarPixel {
  typedef Type T;
};

#include "image/kernels/blurKernel.cu"
#include "image/kernels/blurKernelSmallSupport.cu"
#include "image/kernels/unrolledGaussianKernels.cu"

namespace VideoStitch {
namespace Image {

namespace {
template <typename T>
void swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}
}  // namespace

template <typename T>
Status boxBlur1DNoWrap(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t width, std::size_t height,
                       unsigned radius, unsigned blockSize, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), 1, 1);
  if ((std::size_t)radius >= height) {
    blur1DKernelNoWrapHugeRadius<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                   (unsigned)height, radius);
  } else if ((std::size_t)(2 * radius) >= height) {
    blur1DKernelNoWrapLargeRadius<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                    (unsigned)height, radius);
  } else if (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= radius) {  // if radius is short enough for blurColumnsKernel
    dim3 blocks((unsigned)Cuda::ceilDiv(width, COLUMNS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    blurColumnsKernelNoWrap<T><<<blocks, threads, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                               (unsigned)height, (unsigned)width, radius);
  } else {
    blur1DKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                         (unsigned)height, radius);
  }
  return CUDA_STATUS;
}

template <typename T>
Status boxBlur1DWrap(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t width, std::size_t height,
                     unsigned radius, unsigned blockSize, GPU::Stream stream) {
  if ((std::size_t)(2 * radius) >= height) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(height / 2 - 1);
  }

  if (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= radius) {  // if radius is short enough for blurColumnsKernel
    dim3 blocks((unsigned)Cuda::ceilDiv(width, COLUMNS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    blurColumnsKernelWrap<<<blocks, threads, 0, stream.get()>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                (unsigned)height, (unsigned)width, radius);
  } else {
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), 1, 1);
    blur1DKernelWrap<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                             (unsigned)height, radius);
  }
  return CUDA_STATUS;
}

template Status boxBlur1DNoWrap(GPU::Buffer<float> dst, GPU::Buffer<const float> src, std::size_t width,
                                std::size_t height, unsigned radius, unsigned blockSize, GPU::Stream stream);

template Status boxBlur1DNoWrap(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, std::size_t width,
                                std::size_t height, unsigned radius, unsigned blockSize, GPU::Stream stream);
template Status boxBlur1DNoWrap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                                std::size_t height, unsigned radius, unsigned blockSize, GPU::Stream stream);

template Status boxBlur1DWrap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, unsigned radius, unsigned blockSize, GPU::Stream stream);

template <typename T>
Status gaussianBlur2D(GPU::Buffer<T> dst, GPU::Buffer<const T> src, GPU::Buffer<T> work, std::size_t width,
                      std::size_t height, unsigned radius, unsigned passes, bool wrap, GPU::Stream stream) {
  assert(passes > 0);
  const unsigned blockSize = RGBA_BOX_BLUR_1D_BLOCK_SIZE;
  // First pass is from src to work;
  PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(work, src, width, height, radius, blockSize, stream));
  // Other passes ping-pong between work buffers.
  GPU::Buffer<T> srcBuf = work;
  GPU::Buffer<T> dstBuf = dst;
  for (unsigned i = 1; i < passes; ++i) {
    PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(dstBuf, srcBuf.as_const(), width, height, radius, blockSize, stream));
    swap(dstBuf, srcBuf);
  }
  // transpose
  PROPAGATE_FAILURE_STATUS(transpose(dstBuf.get().raw(), srcBuf.get().raw(), width, height, stream));
  swap(dstBuf, srcBuf);
  if (wrap) {
    for (unsigned i = 0; i < passes; ++i) {
      PROPAGATE_FAILURE_STATUS(boxBlur1DWrap(dstBuf, srcBuf.as_const(), height, width, radius, blockSize, stream));
      swap(dstBuf, srcBuf);
    }
  } else {
    for (unsigned i = 0; i < passes; ++i) {
      PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(dstBuf, srcBuf.as_const(), height, width, radius, blockSize, stream));
      swap(dstBuf, srcBuf);
    }
  }
  PROPAGATE_FAILURE_STATUS(transpose(dstBuf.get().raw(), srcBuf.get().raw(), height, width, stream));
  // There are (passes - 1) swaps, then the transpose swap, then passes swaps.
  // i.e. 2 * passes swaps. So overall srcBuf ad dstBuff are unchanged from their first state.
  assert(dstBuf == dst);
  return CUDA_STATUS;
}

template Status gaussianBlur2D(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src,
                               GPU::Buffer<unsigned char> work, std::size_t width, std::size_t height, unsigned radius,
                               unsigned passes, bool wrap, GPU::Stream stream);
template Status gaussianBlur2D(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, GPU::Buffer<float2> work,
                               std::size_t width, std::size_t height, unsigned radius, unsigned passes, bool wrap,
                               GPU::Stream stream);

Status boxBlurColumnsWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                                 std::size_t height, unsigned radius, GPU::Stream stream) {
  if ((std::size_t)(2 * radius) >= height) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(height / 2 - 1);
  }
  if (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= radius) {  // if radius is short enough for blurColumnsKernel
    dim3 blocks((unsigned)Cuda::ceilDiv(width, COLUMNS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    blurColumnsKernelWrap<uint32_t><<<blocks, threads, 0, stream.get()>>>(
        dst.get().raw(), src.get().raw(), (unsigned)width, (unsigned)height, (unsigned)width, radius);
  } else {
    dim3 dimBlock(RGBA_BOX_BLUR_1D_BLOCK_SIZE, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), 1, 1);
    blur1DKernelWrap<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                             (unsigned)height, radius);
  }
  return CUDA_STATUS;
}

Status boxBlurColumnsNoWrapRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                                   std::size_t height, unsigned radius, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(RGBA_BOX_BLUR_1D_BLOCK_SIZE, 1, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), 1, 1);
  if ((std::size_t)radius >= height) {
    blur1DKernelNoWrapHugeRadius<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                   (unsigned)height, radius);
  } else if ((std::size_t)(2 * radius) >= height) {
    blur1DKernelNoWrapLargeRadius<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                    (unsigned)height, radius);
  } else if (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= radius) {  // if radius is short enough for blurColumnsKernel
    dim3 blocks((unsigned)Cuda::ceilDiv(width, COLUMNS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    blurColumnsKernelNoWrap<uint32_t><<<blocks, threads, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                                      (unsigned)height, (unsigned)width, radius);
  } else {
    blur1DKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(dst.get().raw(), src.get().raw(), (unsigned)width,
                                                         (unsigned)height, radius);
  }
  return CUDA_STATUS;
}

Status boxBlurRowsRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                          std::size_t height, unsigned radius, GPU::Stream stream, bool wrap) {
  dim3 blocks((unsigned)Cuda::ceilDiv(width, (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)),
              (unsigned)Cuda::ceilDiv(height, ROWS_BLOCKDIM_Y));
  dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
  if ((std::size_t)(2 * radius) >= width) {
    // the blur takes the whole buffer for all pixels since the stencil is larger than the patchlet,
    // so just resize the stencil
    radius = (unsigned)(width / 2 - 1);
  }
  if (wrap) {
    blurRowsKernelWrap<<<blocks, threads, 0, stream.get()>>>(dst.get().raw(), src.get().raw(), width, height, width,
                                                             radius);
  } else {
    blurRowsKernelNoWrap<<<blocks, threads, 0, stream.get()>>>(dst.get().raw(), src.get().raw(), width, height, width,
                                                               radius);
  }
  return CUDA_STATUS;
}

Status gaussianBlur2DRGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work,
                             std::size_t width, std::size_t height, unsigned radius, unsigned passes, bool wrap,
                             GPU::Stream stream) {
  assert(passes > 0);
  // First pass is from src to work;
  PROPAGATE_FAILURE_STATUS(boxBlurColumnsNoWrapRGBA210(work, src, width, height, radius, stream));
  // Other passes ping-pong between work buffers.
  GPU::Buffer<uint32_t> srcBuf = work;
  GPU::Buffer<uint32_t> dstBuf = dst;
  for (unsigned i = 1; i < passes; ++i) {
    PROPAGATE_FAILURE_STATUS(boxBlurColumnsNoWrapRGBA210(dstBuf, srcBuf.as_const(), width, height, radius, stream));
    swap(dstBuf, srcBuf);
  }
  if ((ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= radius) &&
      ((std::size_t)2 * radius < height)) {  // boxBlurRowsRGBA210 works only in this case
    for (unsigned i = 0; i < passes; ++i) {
      PROPAGATE_FAILURE_STATUS(boxBlurRowsRGBA210(dstBuf, srcBuf.as_const(), width, height, radius, stream, wrap));
      swap(dstBuf, srcBuf);
    }
    swap(dstBuf, srcBuf);
    assert(dstBuf == dst);
  } else {
    // transpose
    PROPAGATE_FAILURE_STATUS(transpose(dstBuf.get().raw(), srcBuf.get().raw(), width, height, stream));
    swap(dstBuf, srcBuf);
    if (wrap) {
      for (unsigned i = 0; i < passes; ++i) {
        PROPAGATE_FAILURE_STATUS(boxBlurColumnsWrapRGBA210(dstBuf, srcBuf.as_const(), height, width, radius, stream));
        swap(dstBuf, srcBuf);
      }
    } else {
      for (unsigned i = 0; i < passes; ++i) {
        PROPAGATE_FAILURE_STATUS(boxBlurColumnsNoWrapRGBA210(dstBuf, srcBuf.as_const(), height, width, radius, stream));
        swap(dstBuf, srcBuf);
      }
    }
    PROPAGATE_FAILURE_STATUS(transpose(dstBuf.get().raw(), srcBuf.get().raw(), height, width, stream));
    // There are (passes - 1) swaps, then the transpose swap, then passes swaps.
    // i.e. 2 * passes swaps. So overall srcBuf ad dstBuff are unchanged from their first state.
    assert(dstBuf == dst);
  }
  return CUDA_STATUS;
}

Status gaussianBlur1DRGBA210SS(uint32_t* dst, const uint32_t* src, std::size_t width, std::size_t height,
                               unsigned radius, bool wrap, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  // Block organization is as follows for a 5x3 image and dimBlock.x == 3
  //  00 00 00 10 10
  //  01 01 01 11 11
  //  02 02 02 12 12
  // Handle the interior
  if ((unsigned)width > 2 * radius) {
    dim3 dimBlock(RGBA_BOX_BLUR_SS_1D_BLOCK_SIZE, 1, 1);
    dim3 dimGrid((unsigned)Cuda::ceilDiv(width - 2 * radius, dimBlock.x), (unsigned)height, 1);
    assert(2 * radius < dimBlock.x);
    switch (radius) {
      case 1:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel1>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      case 2:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel2>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      case 3:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel3>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      case 4:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel4>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      case 5:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel5>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      case 6:
        gaussianBlur1DRGBA210SSKernelInterior<unrolledGaussianKernel6>
            <<<dimGrid, dimBlock, 16 * (dimBlock.x + 2 * radius), stream>>>(dst, src, (unsigned)width, (unsigned)height,
                                                                            radius);
        break;
      default:
        assert(false);
        break;
    }
  }
  // There are exactly radius pixels on each border (left and right) + radius pixels before and after them.
  assert(4 * radius <= 32);
  dim3 dimBlock(4 * radius, 1, 1);
  dim3 dimGrid(1, (unsigned)height, 1);
  if (wrap) {
    switch (radius) {
      case 1:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel1>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 2:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel2>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 3:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel3>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 4:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel4>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 5:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel5>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 6:
        gaussianBlur1DRGBA210SSKernelWrap<unrolledGaussianKernel6>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      default:
        assert(false);
        break;
    }
  } else {
    switch (radius) {
      case 1:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel1>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 2:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel2>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 3:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel3>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 4:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel4>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 5:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel5>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      case 6:
        gaussianBlur1DRGBA210SSKernelNoWrap<unrolledGaussianKernel6>
            <<<dimGrid, dimBlock, 16 * dimBlock.x, stream>>>(dst, src, (unsigned)width, (unsigned)height, radius);
        break;
      default:
        assert(false);
        break;
    }
  }
  return CUDA_STATUS;
}

Status gaussianBlur2DRGBA210SS(uint32_t* buf, uint32_t* work, std::size_t width, std::size_t height, unsigned radius,
                               bool wrap, GPU::Stream stream) {
  // Vertical pass, never wraps.
  PROPAGATE_FAILURE_STATUS(gaussianBlur1DRGBA210SS(work, buf, width, height, radius, false, stream));
  // transpose
  PROPAGATE_FAILURE_STATUS(transpose(buf, work, width, height, stream));
  PROPAGATE_FAILURE_STATUS(gaussianBlur1DRGBA210SS(work, buf, height, width, radius, wrap, stream));
  return transpose(buf, work, height, width, stream);
}

// TODO_GPU_DEPRECATE
// only used in test currently
Status gaussianBlur2D(GPU::Buffer<unsigned char> buf, GPU::Buffer<unsigned char> work, std::size_t width,
                      std::size_t height, unsigned radius, unsigned passes, bool wrap, unsigned blockSize,
                      GPU::Stream stream) {
  // Avoid copy: force even passes
  assert((passes & 1) == 0);
  for (unsigned i = 0; i < passes / 2; ++i) {
    PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(work, buf.as_const(), width, height, radius, blockSize, stream));
    PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(buf, work.as_const(), width, height, radius, blockSize, stream));
  }
  // transpose
  PROPAGATE_FAILURE_STATUS(transpose(work.get().raw(), buf.as_const().get().raw(), width, height, stream));
  for (unsigned i = 0; i < passes / 2; ++i) {
    if (wrap) {
      PROPAGATE_FAILURE_STATUS(boxBlur1DWrap(buf, work.as_const(), height, width, radius, blockSize, stream));
      PROPAGATE_FAILURE_STATUS(boxBlur1DWrap(work, buf.as_const(), height, width, radius, blockSize, stream));
    } else {
      PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(buf, work.as_const(), height, width, radius, blockSize, stream));
      PROPAGATE_FAILURE_STATUS(boxBlur1DNoWrap(work, buf.as_const(), height, width, radius, blockSize, stream));
    }
  }
  return transpose(buf.get().raw(), work.as_const().get().raw(), height, width, stream);
}

Status gaussianBlur2DRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work,
                          std::size_t width, std::size_t height, unsigned /*radius*/, unsigned /*passes*/, bool wrap,
                          GPU::Stream stream) {
  uint32_t* h_Kernel = (uint32_t*)malloc((2 * KERNEL_RADIUS + 1) * sizeof(uint32_t));
  h_Kernel[0] = 1;
  h_Kernel[1] = 4;
  h_Kernel[2] = 6;
  h_Kernel[3] = 4;
  h_Kernel[4] = 1;
  setConvolutionKernel(h_Kernel);

  {
    dim3 blocks((unsigned)Cuda::ceilDiv(width, ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, ROWS_BLOCKDIM_Y));
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    if (wrap) {
      convolutionRowsKernel<true><<<blocks, threads, 0, stream.get()>>>(
          work.get().raw(), src.get().raw(), (unsigned)width, (unsigned)height, (unsigned)width);
    } else {
      convolutionRowsKernel<false><<<blocks, threads, 0, stream.get()>>>(
          work.get().raw(), src.get().raw(), (unsigned)width, (unsigned)height, (unsigned)width);
    }
  }

  {
    dim3 blocks((unsigned)Cuda::ceilDiv(width, COLUMNS_BLOCKDIM_X),
                (unsigned)Cuda::ceilDiv(height, COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads, 0, stream.get()>>>(dst.get().raw(), work.get().raw(), (unsigned)width,
                                                                   (unsigned)height, (unsigned)width);
  }

  free(h_Kernel);

  return CUDA_STATUS;
}

}  // namespace Image
}  // namespace VideoStitch
