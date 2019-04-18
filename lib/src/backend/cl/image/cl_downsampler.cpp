// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/downsampler.hpp"

#include "gpu/2dBuffer.hpp"

#include "../kernel.hpp"
#include "../deviceBuffer2D.hpp"

namespace {
#include "downsampler.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(downsampler, true);

namespace VideoStitch {
namespace Image {

Status downsampleRGBASurf2x(GPU::Surface& /*dst*/, const GPU::Surface& /*src*/, unsigned /*dstWidth*/,
                            unsigned /*dstHeight*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Downsampling RGBA surfaces not implemented"};
}

Status downsampleRGBA(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleRGBAKernel))
                      .setup2D(stream, (unsigned)out.getWidth(), (unsigned)out.getHeight());
  return kernel2D.enqueueWithKernelArgs(out.get(), (unsigned)out.getPitch(), in.get(), (unsigned)in.getPitch(),
                                        (unsigned)in.getWidth(), (unsigned)in.getHeight(), (unsigned)factor);
}

Status downsampleRGB(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleRGBKernel))
                      .setup2D(stream, (unsigned)out.getWidth(), (unsigned)out.getHeight());
  return kernel2D.enqueueWithKernelArgs(out.get(), (unsigned)out.getPitch(), in.get(), (unsigned)in.getPitch(),
                                        (unsigned)in.getWidth(), (unsigned)in.getHeight(), (unsigned)factor);
}

Status downsampleYUV422(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleYUV422Kernel))
                      .setup2D(stream, (unsigned)out.getWidth(), (unsigned)out.getHeight());
  return kernel2D.enqueueWithKernelArgs(out.get(), (unsigned)out.getPitch(), in.get(), (unsigned)in.getPitch(),
                                        (unsigned)in.getWidth(), (unsigned)in.getHeight(), (unsigned)factor);
}

Status downsampleYUV422P10(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                           GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleYUV422P10Kernel))
                      .setup2D(stream, (unsigned)yOut.getWidth(), (unsigned)yOut.getHeight());
  return kernel2D.enqueueWithKernelArgs(yOut.get(), (unsigned)yOut.getPitch(), uOut.get(), (unsigned)uOut.getPitch(),
                                        vOut.get(), (unsigned)vOut.getPitch(), yIn.get(), (unsigned)yIn.getPitch(),
                                        uIn.get(), (unsigned)uIn.getPitch(), vIn.get(), (unsigned)vIn.getPitch(),
                                        (unsigned)yIn.getWidth(), (unsigned)yIn.getHeight(), (unsigned)factor);
}

Status downsampleYV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                      GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleYV12Kernel))
                      .setup2D(stream, (unsigned)yOut.getWidth(), (unsigned)yOut.getHeight());
  return kernel2D.enqueueWithKernelArgs(yOut.get(), (unsigned)yOut.getPitch(), uOut.get(), (unsigned)uOut.getPitch(),
                                        vOut.get(), (unsigned)vOut.getPitch(), yIn.get(), (unsigned)yIn.getPitch(),
                                        uIn.get(), (unsigned)uIn.getPitch(), vIn.get(), (unsigned)vIn.getPitch(),
                                        (unsigned)yIn.getWidth(), (unsigned)yIn.getHeight(), (unsigned)factor);
}

Status downsampleNV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uvIn, GPU::Buffer2D& yOut, GPU::Buffer2D& uvOut,
                      GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(downsampler), KERNEL_STR(downsampleNV12Kernel))
                      .setup2D(stream, (unsigned)yOut.getWidth(), (unsigned)yOut.getHeight());
  return kernel2D.enqueueWithKernelArgs(yOut.get(), (unsigned)yOut.getPitch(), uvOut.get(), (unsigned)uvOut.getPitch(),
                                        yIn.get(), (unsigned)yIn.getPitch(), uvIn.get(), (unsigned)uvIn.getPitch(),
                                        (unsigned)yIn.getWidth(), (unsigned)yIn.getHeight(), (unsigned)factor);
}
}  // namespace Image
}  // namespace VideoStitch
