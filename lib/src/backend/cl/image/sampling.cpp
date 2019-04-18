// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../../../gpu/image/sampling.hpp"

#include "../kernel.hpp"

namespace {
#include "sampling.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(sampling, true);

#define BLOCK_SIZE 16

namespace VideoStitch {
namespace Image {

/**
 * Subsample a buffer by a factor of two, picking the topleft value for every 2x2 pixels blocks.
 * WARNING: no antialiasing filter ! Blur first !
 * @param dst subsampled buffer, size (srcWidth / 2) * (srcHeight / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status subsample22(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t srcWidth, std::size_t srcHeight,
                   GPU::Stream stream) {
  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RegularKernel))
                        .setup2D(stream, (unsigned)dstWidth, (unsigned)dstHeight);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth,
                                               (unsigned)dstHeight));
  }
  // right boundary
  if (srcWidth & 1) {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RightBoundaryKernel))
                        .setup1D(stream, (unsigned)dstHeight);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth,
                                               (unsigned)dstHeight));
  }
  // bottom boundary
  if (srcHeight & 1) {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RightBoundaryKernel))
                        .setup1D(stream, (unsigned)dstWidth);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth,
                                               (unsigned)dstHeight));
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy of the last element
    return CL_ERROR(clEnqueueCopyBuffer(stream.get(), src.get(), dst.get(), srcHeight * srcWidth - 1,
                                        dstWidth * dstHeight - 1, sizeof(T), 0, nullptr, nullptr));
  }
  return Status::OK();
}

/**
 * Subsample a buffer by a factor of two, picking the topleft value for every 2x2 pixels blocks.
 * WARNING: no antialiasing filter ! Blur first !
 * @param dst subsampled buffer, size (srcWidth / 2) * (srcHeight / 2). Pixels in RGB210.
 * @param src subsampled buffer, size srcWidth * srcHeight. Pixel in RGBA.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param stream Cuda stream to run in.
 */
Status subsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t srcWidth,
                       std::size_t srcHeight, GPU::Stream stream) {
  std::size_t dstWidth = (srcWidth + 1) / 2;
  std::size_t dstHeight = (srcHeight + 1) / 2;
  // interior
  {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RGBARegularKernel))
                        .setup2D(stream, (unsigned)dstWidth, (unsigned)dstHeight);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth));
  }
  // right boundary
  if (srcWidth & 1) {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RGBARightBoundaryKernel))
                        .setup1D(stream, (unsigned)dstHeight);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth));
  }
  // bottom boundary
  if (srcHeight & 1) {
    auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(subsample22RGBABottomBoundaryKernel))
                        .setup1D(stream, (unsigned)dstWidth);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)srcWidth, (unsigned)srcHeight, (unsigned)dstWidth));
  }
  if ((srcWidth & 1) && (srcHeight & 1)) {
    // simple copy of the last element
    return CL_ERROR(clEnqueueCopyBuffer(stream.get(), src.get(), dst.get(), srcHeight * srcWidth - 1,
                                        dstWidth * dstHeight - 1, sizeof(uint32_t), 0, nullptr, nullptr));
  }
  return Status::OK();
}

/**
 * Upsamples a buffer.
 * @param dst subsampled buffer, size dstWidth * dstHeight.
 * @param src subsampled buffer, size (dstWidth / 2) * (dstHeight / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param stream Cuda stream to run in.
 */
template <typename T>
Status upsample22(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t dstWidth, std::size_t dstHeight, bool wrap,
                  GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(upsample22KernelScalar))
                      .setup2D(stream, (unsigned)srcWidth, (unsigned)srcHeight, BLOCK_SIZE);
  return kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)dstWidth, (unsigned)dstHeight, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (int)wrap);
}

/**
 * Upsamples an image in RGB210.
 * @param dst subsampled buffer, size dstWidth * dstHeight.
 * @param src subsampled buffer, size (dstWidth / 2) * (dstHeight / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param stream Cuda stream to run in.
 */
Status upsample22RGBA210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                         std::size_t dstHeight, bool wrap, GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(upsample22KernelRGB210))
                      .setup2D(stream, (unsigned)srcWidth, (unsigned)srcHeight, BLOCK_SIZE);
  return kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)dstWidth, (unsigned)dstHeight, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (int)wrap);
}

/**
 * Upsamples an image in RGBA.
 * @param dst subsampled buffer, size dstWidth * dstHeight.
 * @param src subsampled buffer, size (dstWidth / 2) * (dstHeight / 2).
 * @param dstWidth Destination width.
 * @param dstHeight Destination height.
 * @param blockSize Cuda block size (effective size is blockSize * blockSize)
 * @param stream Cuda stream to run in.
 */
Status upsample22RGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t dstWidth,
                      std::size_t dstHeight, bool wrap, GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  auto kernel2D = GPU::Kernel::get(PROGRAM(sampling), KERNEL_STR(upsample22KernelRGBA))
                      .setup2D(stream, (unsigned)srcWidth, (unsigned)srcHeight, BLOCK_SIZE);
  return kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)dstWidth, (unsigned)dstHeight, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (int)wrap);
}

/**
 * Subsamples the given mask by a factor of two. For each 2x2 pixel block, the output pixel is masked out if any of the
 * input pixels is masked out (i.e. any pixel has value 1).
 * @param dst subsampled buffer, size (srcWidth / 2) * (srcHeight / 2).
 * @param src subsampled buffer, size srcWidth * srcHeight.
 * @param srcWidth Source width.
 * @param srcHeight Source height.
 * @param blockSize Cuda block size (effective size is blockSize * blockSize)
 * @param stream Cuda stream to run in.
 */
Status subsampleMask22(GPU::Buffer<unsigned char> /*dst*/, GPU::Buffer<const unsigned char> /*src*/,
                       std::size_t /*srcWidth*/, std::size_t /*srcHeight*/, unsigned int /*blockSize*/,
                       GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "Masked subsampling not implemented in OpenCL backend"};
}

#include "../../common/sampling.inst"

}  // namespace Image
}  // namespace VideoStitch
