// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <gpu/core1/mergerKernel.hpp>
#include <core1/imageMapping.hpp>
#include "libvideostitch/panoDef.hpp"

#include "../kernel.hpp"

namespace {
#include "mergerKernel.xxd"

INDIRECT_REGISTER_OPENCL_PROGRAM(mergerKernel, true);

}  // namespace

namespace VideoStitch {
namespace Core {

Status countInputs(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                   const Core::ImageMapping& fromIm, GPU::Stream stream) {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  std::string kernelName;
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    kernelName = KERNEL_STR(countInputsKernelWrap);
  } else {
    kernelName = KERNEL_STR(countInputsKernelNoWrap);
  }

  auto kernel2D =
      GPU::Kernel::get(PROGRAM(mergerKernel), kernelName)
          .setup2D(stream, (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight());
  return kernel2D.enqueueWithKernelArgs(
      panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
      (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
      (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
}

Status colorMap(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream) {
  const int64_t size = pano.getWidth() * pano.getHeight();
  auto kernel2D = GPU::Kernel::get(PROGRAM(mergerKernel), KERNEL_STR(colormapKernel)).setup1D(stream, (unsigned)size);
  return kernel2D.enqueueWithKernelArgs(panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)size);
}

Status stitchingError(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                      const ImageMapping& fromIm, GPU::Stream stream) {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  std::string kernelName;
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    kernelName = KERNEL_STR(stitchingErrorKernelWrap);
  } else {
    kernelName = KERNEL_STR(stitchingErrorKernelNoWrap);
  }

  auto kernel2D =
      GPU::Kernel::get(PROGRAM(mergerKernel), kernelName)
          .setup2D(stream, (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight());
  return kernel2D.enqueueWithKernelArgs(
      panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
      (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
      (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
}

Status exposureDiffRGB(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                       const ImageMapping& fromIm, GPU::Stream stream) {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  std::string kernelName;
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    kernelName = KERNEL_STR(exposureErrorRGBKernelWrap);
  } else {
    kernelName = KERNEL_STR(exposureErrorRGBKernelNoWrap);
  }

  auto kernel2D =
      GPU::Kernel::get(PROGRAM(mergerKernel), kernelName)
          .setup2D(stream, (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight());
  return kernel2D.enqueueWithKernelArgs(
      panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
      (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
      (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
}

Status amplitude(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream) {
  const int64_t size = pano.getWidth() * pano.getHeight();
  auto kernel1D = GPU::Kernel::get(PROGRAM(mergerKernel), KERNEL_STR(amplitudeKernel)).setup1D(stream, (unsigned)size);
  return kernel1D.enqueueWithKernelArgs(panoDevOut.get(), 0, (3 * 256 * 256), (unsigned)pano.getWidth(),
                                        (unsigned)size);
}

Status disregardNoDiffArea(const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, GPU::Stream stream) {
  const int64_t size = pano.getWidth() * pano.getHeight();
  auto kernel1D =
      GPU::Kernel::get(PROGRAM(mergerKernel), KERNEL_STR(maskOutSingleInput)).setup1D(stream, (unsigned)size);
  return kernel1D.enqueueWithKernelArgs(panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)size);
}

Status checkerMerge(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                    const ImageMapping& fromIm, unsigned checkerSize, GPU::Stream stream) {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  std::string kernelName;
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    kernelName = KERNEL_STR(checkerInsertKernelWrap);
  } else {
    kernelName = KERNEL_STR(checkerInsertKernelNoWrap);
  }

  auto kernel2D =
      GPU::Kernel::get(PROGRAM(mergerKernel), kernelName)
          .setup2D(stream, (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight());
  return kernel2D.enqueueWithKernelArgs(
      panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
      (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
      (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top(), checkerSize);
}

Status noblend(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
               const ImageMapping& fromIm, GPU::Stream stream) {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  std::string kernelName;
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    kernelName = KERNEL_STR(noblendKernelWrap);
  } else {
    kernelName = KERNEL_STR(noblendKernelNoWrap);
  }

  auto kernel2D =
      GPU::Kernel::get(PROGRAM(mergerKernel), kernelName)
          .setup2D(stream, (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight());
  return kernel2D.enqueueWithKernelArgs(
      panoDevOut.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
      (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
      (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
};

}  // namespace Core
}  // namespace VideoStitch
