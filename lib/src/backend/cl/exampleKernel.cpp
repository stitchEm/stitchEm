// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "kernel.hpp"

namespace VideoStitch {
namespace Core {

namespace {
#include "exampleKernel.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(exampleKernel, false);

Status callDummyKernel(GPU::Buffer<float> outputBuff, const GPU::Buffer<const float>& inputBuff,
                       unsigned int nbElements, float mult, GPU::Stream stream) {
  auto bidonCompute = GPU::Kernel::get(PROGRAM(exampleKernel), KERNEL_STR(vecAddDummy)).setup1D(stream, 64, 16);
  return bidonCompute.enqueueWithKernelArgs(outputBuff, inputBuff, nbElements, mult);
}

}  // namespace Core
}  // namespace VideoStitch
