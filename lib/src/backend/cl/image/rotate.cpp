// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imageOps.hpp"

#include "../kernel.hpp"

namespace {
#include "rotate.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(rotate, true);

namespace VideoStitch {
namespace Image {

Status flip(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(rotate), KERNEL_STR(flipKernel)).setup2D(stream, (unsigned)length, (unsigned)length);
  FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)length));
  return Status::OK();
}

Status flop(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(rotate), KERNEL_STR(flopKernel)).setup2D(stream, (unsigned)length, (unsigned)length);
  FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)length));
  return Status::OK();
}

Status rotate(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(rotate), KERNEL_STR(rotateKernel)).setup2D(stream, (unsigned)length, (unsigned)length);
  FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)length));
  return Status::OK();
}

Status rotateLeft(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(rotate), KERNEL_STR(rotateLeftKernel))
                      .setup2D(stream, (unsigned)length, (unsigned)length);
  FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)length));
  return Status::OK();
}

Status rotateRight(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, std::size_t length, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(rotate), KERNEL_STR(rotateRightKernel))
                      .setup2D(stream, (unsigned)length, (unsigned)length);
  FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)length));
  return Status::OK();
}

}  // namespace Image
}  // namespace VideoStitch
