// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/render/render.hpp"

#include "../cl_error.hpp"
#include "../context.hpp"
#include "../kernel.hpp"
#include "../surface.hpp"

#ifdef __APPLE__

namespace {
#include "render.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(render, true);
}  // namespace

#endif  // __APPLE__

namespace VideoStitch {
namespace Render {

#ifdef __APPLE__

Status fillBuffer(GPU::Buffer<uint32_t> dst, uint32_t value, size_t width, size_t height, GPU::Stream stream) {
  auto kernel =
      GPU::Kernel::get(PROGRAM(render), KERNEL_STR(memsetToValue)).setup1D(stream, (unsigned)(width * height));
  return kernel.enqueueWithKernelArgs(dst.get(), value, (unsigned)dst.numElements());
}

#else  // __APPLE__

Status fillBuffer(GPU::Buffer<uint32_t> dst, uint32_t value, size_t width, size_t height, GPU::Stream stream) {
  return CL_ERROR(clEnqueueFillBuffer(stream.get(), dst.get().raw(), &value, sizeof(uint32_t),
                                      0,  // offset
                                      width * height * sizeof(uint32_t), 0, nullptr, nullptr));
}

#endif  // __APPLE__

}  // namespace Render
}  // namespace VideoStitch
