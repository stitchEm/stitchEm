// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imgInsert.hpp"

#include "../kernel.hpp"

namespace VideoStitch {
namespace Image {

namespace {
#include "imgInsert.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(imgInsert, true);

namespace {

std::string kernelNameFromArguments(GPU::Buffer<const unsigned char> mask, bool hWrap, bool vWrap, bool tenBits) {
  std::string kernelName;
  if (tenBits) {
    if (mask.wasAllocated()) {
      if (vWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedVWrap10bit);
      } else if (hWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedHWrap10bit);
      } else {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedNoWrap10bit);
      }
    } else {
      if (vWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelVWrap10bit);
      } else if (hWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelHWrap10bit);
      } else {
        kernelName = KERNEL_STR(imgInsertIntoKernelNoWrap10bit);
      }
    }
  } else {
    if (mask.wasAllocated()) {
      if (vWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedVWrap);
      } else if (hWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedHWrap);
      } else {
        kernelName = KERNEL_STR(imgInsertIntoKernelMaskedNoWrap);
      }
    } else {
      if (vWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelVWrap);
      } else if (hWrap) {
        kernelName = KERNEL_STR(imgInsertIntoKernelHWrap);
      } else {
        kernelName = KERNEL_STR(imgInsertIntoKernelNoWrap);
      }
    }
  }
  return kernelName;
}
}  // namespace

Status imgInsertInto(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                     GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                     std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap, bool vWrap,
                     GPU::Stream stream) {
  if (hWrap && vWrap) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Cannot wrap horizontally and vertically"};
  }

  const auto kernelName = kernelNameFromArguments(mask, hWrap, vWrap, false);
  auto kernel =
      GPU::Kernel::get(PROGRAM(imgInsert), kernelName).setup2D(stream, (unsigned)srcWidth, (unsigned)srcHeight);
  if (mask.wasAllocated()) {
    return kernel.enqueueWithKernelArgs(dst, (unsigned)dstWidth, (unsigned)dstHeight, src, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (unsigned)offsetX, (unsigned)offsetY, mask);
  } else {
    return kernel.enqueueWithKernelArgs(dst, (unsigned)dstWidth, (unsigned)dstHeight, src, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (unsigned)offsetX, (unsigned)offsetY);
  }
}

Status imgInsertInto10bit(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                          GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight,
                          std::size_t offsetX, std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap,
                          bool vWrap, GPU::Stream stream) {
  if (hWrap && vWrap) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Cannot wrap horizontally and vertically"};
  }

  const auto kernelName = kernelNameFromArguments(mask, hWrap, vWrap, true);
  auto kernel =
      GPU::Kernel::get(PROGRAM(imgInsert), kernelName).setup2D(stream, (unsigned)srcWidth, (unsigned)srcHeight);
  if (mask.wasAllocated()) {
    return kernel.enqueueWithKernelArgs(dst, (unsigned)dstWidth, (unsigned)dstHeight, src, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (unsigned)offsetX, (unsigned)offsetY, mask);
  } else {
    return kernel.enqueueWithKernelArgs(dst, (unsigned)dstWidth, (unsigned)dstHeight, src, (unsigned)srcWidth,
                                        (unsigned)srcHeight, (unsigned)offsetX, (unsigned)offsetY);
  }
}
}  // namespace Image
}  // namespace VideoStitch
