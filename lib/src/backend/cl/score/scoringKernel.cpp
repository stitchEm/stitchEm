// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <score/scoringProcessor.hpp>

#include "../kernel.hpp"

namespace {
#include "scoringKernel.xxd"
}
INDIRECT_REGISTER_OPENCL_PROGRAM(scoringKernel, true);

namespace VideoStitch {
namespace Image {

Status splitNoBlendImageMergerChannel(GPU::Buffer<float> dest_r, GPU::Buffer<float> dest_g,
                                      GPU::Buffer<unsigned char> dest_b, GPU::Buffer<const uint32_t> source,
                                      const unsigned width, const unsigned height, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(scoringKernel), KERNEL_STR(splitNoBlendImageMergerChannelsKernel))
                      .setup2D(stream, width, height);
  return kernel2D.enqueueWithKernelArgs(dest_r.get(), dest_g.get(), dest_b.get(), source.get(), width, height);
}

}  // namespace Image
}  // namespace VideoStitch
