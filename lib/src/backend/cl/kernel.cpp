// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "kernel.hpp"

namespace VideoStitch {
namespace GPU {

// TODO ceil size?
static const size_t CEIL = 16;

static size_t ceilKernelSize(size_t v, size_t d) {
  const size_t res = v / d;
  const size_t group = res + (v - res * d > 0);  // add one if the remainder is nonzero
  return group * d;
}

Kernel Kernel::get(std::string programName, std::string kernelName) {
  const auto& potContext = GPU::getContext();
  assert(potContext.ok());
  CLKernel& potCLKernel = potContext.value().getKernel(programName, kernelName);
  return Kernel{potCLKernel};
}

KernelExecution<1> Kernel::setup1D(GPU::Stream stream, unsigned totalSize) const {
  size_t global{ceilKernelSize((size_t)totalSize, CEIL)};
  size_t local{0};
  return KernelExecution<1>{kernel, stream, {{global}}, {{local}}};
}

KernelExecution<1> Kernel::setup1D(GPU::Stream stream, unsigned totalSize, unsigned blockSize) const {
  size_t global1D{ceilKernelSize((size_t)totalSize, blockSize)};
  size_t local1D{blockSize};
  return KernelExecution<1>{kernel, stream, {{global1D}}, {{local1D}}};
}

KernelExecution<2> Kernel::setup2D(GPU::Stream stream, unsigned totalWidth, unsigned totalHeight) const {
  std::array<size_t, 2> global2D{{ceilKernelSize((size_t)totalWidth, CEIL), ceilKernelSize((size_t)totalHeight, CEIL)}};
  std::array<size_t, 2> local2D{{0, 0}};
  return KernelExecution<2>{kernel, stream, global2D, local2D};
}

KernelExecution<2> Kernel::setup2D(GPU::Stream stream, unsigned totalWidth, unsigned totalHeight, unsigned blockSizeX,
                                   unsigned blockSizeY) const {
  std::array<size_t, 2> global2D{
      {ceilKernelSize((size_t)totalWidth, blockSizeX), ceilKernelSize((size_t)totalHeight, blockSizeY)}};
  std::array<size_t, 2> local2D{{blockSizeX, blockSizeY}};
  return KernelExecution<2>{kernel, stream, global2D, local2D};
}

}  // namespace GPU
}  // namespace VideoStitch
