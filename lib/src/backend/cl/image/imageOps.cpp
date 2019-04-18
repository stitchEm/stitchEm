// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imageOps.hpp"

#include "../kernel.hpp"

namespace {
#include "imageOps.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(imageOps, true);

namespace VideoStitch {
namespace Image {

/**
 * Subtract an image from another: dst -= toSubtract.
 * Alpha is handled the following way:
 *  a   b   (a - b)
 *  1   1      1       normal case
 *  0   0      0       obviously
 *  1   0      1       value is a
 *  0   1      0       This is the hard part. I chose 0. value is unimportant.
 * @param dst Left hand side.
 * @param toSubtract Right hand side
 * @param size Size in pixels of both images.
 * @param stream CUDA stream for the operation
 */
template <typename T>
Status subtractRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toSubtract, std::size_t size, GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(subtractKernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toSubtract, (unsigned)size));
  return Status::OK();
}

Status subtract(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(subtractRGBKernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toSubtract, (unsigned)size));
  return Status::OK();
}

/**
 * Add an image to another: dst += toAdd.
 * Alpha is handled the following way:
 *  a   b   (a + b)
 *  1   1      1       normal case
 *  0   0      0       obviously
 *  1   0      1       value is a
 *  0   1      0       value is unimportant.
 *
 * WARNING: it means that addition is non-commutative!
 * @param dst Left hand side.
 * @param toAdd Right hand side
 * @param size Size in pixels of both images.
 * @param stream CUDA stream for the operation
 */
template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd, std::size_t size, GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(addKernelEx)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd, (unsigned)size));
  return Status::OK();
}

template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd0, GPU::Buffer<const T> toAdd1, std::size_t size,
              GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(addKernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd0, toAdd1, (unsigned)size));
  return Status::OK();
}

Status addRGB210(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const uint32_t> /*toAdd*/, std::size_t /*size*/,
                 GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "Pixel addition not implemented in OpenCL backend"};
}

Status add10(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(add10Kernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd, (unsigned)size));
  return Status::OK();
}

Status add10n8(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(add10n8Kernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd, (unsigned)size));
  return Status::OK();
}

Status add10n8Clamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size,
                    GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(add10n8ClampKernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd, (unsigned)size));
  return Status::OK();
}

Status addClamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream) {
  auto kernel1D = GPU::Kernel::get(PROGRAM(imageOps), KERNEL_STR(addClampKernel)).setup1D(stream, (unsigned)size);
  FAIL_RETURN(kernel1D.enqueueWithKernelArgs(dst, toAdd, (unsigned)size));
  return Status::OK();
}

Status addRGB210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> /*toAdd0*/,
                 GPU::Buffer<const uint32_t> /*toAdd1*/, std::size_t /*size*/, GPU::Stream /*stream*/) {
  return {Origin::Stitcher, ErrType::UnsupportedAction, "RGB210 addition not implemented in OpenCL backend"};
}

/**
 * Note:
 *  a   b     (a - b) + b
 *  1   1         1       value is a
 *  0   0         0       obviously
 *  1   0         1       value is a
 *  0   1         0       consistent.
 */

#include "../../common/image/imageOps.inst"
}  // namespace Image
}  // namespace VideoStitch
