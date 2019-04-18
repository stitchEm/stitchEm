// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/boundsKernel.hpp"

#include "../context.hpp"
#include "../kernel.hpp"

namespace VideoStitch {
namespace Core {

namespace {
#include "boundsKernel.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(boundsKernel, true);

Status vertOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
              GPU::Buffer<uint32_t> colHasImage, GPU::Stream stream) {
  auto kernel =
      GPU::Kernel::get(PROGRAM(boundsKernel), KERNEL_STR(vertOrKernel)).setup1D(stream, (unsigned)croppedWidth);
  return kernel.enqueueWithKernelArgs(contrib, colHasImage, (unsigned)croppedWidth, (unsigned)croppedHeight);
}

Status horizOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
               GPU::Buffer<uint32_t> rowHasImage, GPU::Stream stream) {
  auto kernel =
      GPU::Kernel::get(PROGRAM(boundsKernel), KERNEL_STR(horizOrKernel)).setup1D(stream, (unsigned)croppedHeight);
  return kernel.enqueueWithKernelArgs(contrib, rowHasImage, (unsigned)croppedWidth, (unsigned)croppedHeight);
}

}  // namespace Core
}  // namespace VideoStitch
