// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/input/checkerBoard.hpp"

#include "../kernel.hpp"

namespace {
#include "checkerBoard.xxd"

INDIRECT_REGISTER_OPENCL_PROGRAM(checkerBoard, true);

}  // namespace

namespace VideoStitch {
namespace Input {

Status overlayCheckerBoard(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, unsigned checkerSize,
                           uint32_t color1, uint32_t color2, uint32_t color3, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(checkerBoard), KERNEL_STR(checkerBoardKernel)).setup2D(stream, width, height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), width, height, checkerSize, color1, color2, color3);
}

}  // namespace Input
}  // namespace VideoStitch
