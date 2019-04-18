// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/grid.hpp"

#include "../kernel.hpp"

namespace {

#include "grid.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(grid, true)

}  // namespace

namespace VideoStitch {
namespace Core {

Status grid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth, uint32_t color,
            uint32_t bgColor, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(grid), KERNEL_STR(gridKernel)).setup2D(stream, width, height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), width, height, size, lineWidth, color, bgColor);
}

Status transparentForegroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t bgColor, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(grid), KERNEL_STR(transparentFGGridKernel)).setup2D(stream, width, height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), width, height, size, lineWidth, bgColor);
}

Status transparentBackgroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t color, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(grid), KERNEL_STR(transparentBGGridKernel)).setup2D(stream, width, height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), width, height, size, lineWidth, color);
}

}  // namespace Core
}  // namespace VideoStitch
