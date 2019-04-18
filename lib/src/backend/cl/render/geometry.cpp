// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/render/geometry.hpp"

#include "../surface.hpp"
#include "../deviceStream.hpp"

#include "../kernel.hpp"

namespace VideoStitch {
namespace Render {

namespace {
#include "geometry.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(geometry, true);

template <>
Status drawLine(GPU::Surface& dst, int64_t width, int64_t height, float aX, float aY, float bX, float bY, float t,
                uint32_t color, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(geometry), KERNEL_STR(lineSourceKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), (unsigned)width, (unsigned)height, aX, aY, bX, bY, t, color);
}
template <>
Status drawLine(GPU::Buffer<uint32_t>& dst, int64_t width, int64_t height, float aX, float aY, float bX, float bY,
                float t, uint32_t color, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(geometry), KERNEL_STR(lineKernel)).setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), (unsigned)width, (unsigned)height, aX, aY, bX, bY, t, color);
}

template <>
Status drawDisk(GPU::Surface& dst, int64_t width, int64_t height, float aX, float aY, float thickness, uint32_t color,
                GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(geometry), KERNEL_STR(diskSourceKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), (unsigned)width, (unsigned)height, (float)aX, (float)aY, thickness,
                                        color);
}
template <>
Status drawDisk(GPU::Buffer<uint32_t>& dst, int64_t width, int64_t height, float aX, float aY, float thickness,
                uint32_t color, GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(geometry), KERNEL_STR(diskKernel)).setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get(), (unsigned)width, (unsigned)height, (float)aX, (float)aY, thickness,
                                        color);
}

enum kernelName { circle, circleTop, circleBottom, circleTopRight, circleBottomRight };
template <kernelName kernel, typename Image>
const char* circleKernelName();

template <>
const char* circleKernelName<circle, GPU::Buffer<uint32_t>>() {
  return KERNEL_STR(circleKernel);
}
template <>
const char* circleKernelName<circle, GPU::Surface>() {
  return KERNEL_STR(circleKernelSource);
}

template <>
const char* circleKernelName<circleTop, GPU::Buffer<uint32_t>>() {
  return KERNEL_STR(circleTKernel);
}
template <>
const char* circleKernelName<circleTop, GPU::Surface>() {
  return KERNEL_STR(circleTKernelSource);
}

template <>
const char* circleKernelName<circleBottom, GPU::Buffer<uint32_t>>() {
  return KERNEL_STR(circleBKernel);
}
template <>
const char* circleKernelName<circleBottom, GPU::Surface>() {
  return KERNEL_STR(circleBKernelSource);
}

template <>
const char* circleKernelName<circleTopRight, GPU::Buffer<uint32_t>>() {
  return KERNEL_STR(circleTRKernel);
}
template <>
const char* circleKernelName<circleTopRight, GPU::Surface>() {
  return KERNEL_STR(circleTRKernelSource);
}

template <>
const char* circleKernelName<circleBottomRight, GPU::Buffer<uint32_t>>() {
  return KERNEL_STR(circleBRKernel);
}
template <>
const char* circleKernelName<circleBottomRight, GPU::Surface>() {
  return KERNEL_STR(circleBRKernelSource);
}

#define CIRCLE_FN(fnName, type)                                                                                \
  template <typename Image>                                                                                    \
  Status fnName(Image& dst, int64_t width, int64_t height, float centerX, float centerY, float innerSqrRadius, \
                float outerSqrRadius, uint32_t color, GPU::Stream stream) {                                    \
    std::string kernelName = circleKernelName<type, Image>();                                                  \
    auto kernel2D =                                                                                            \
        GPU::Kernel::get(PROGRAM(geometry), kernelName).setup2D(stream, (unsigned)width, (unsigned)height);    \
    return kernel2D.enqueueWithKernelArgs(dst.get(), (unsigned)width, (unsigned)height, centerX, centerY,      \
                                          innerSqrRadius, outerSqrRadius, color);                              \
  }                                                                                                            \
                                                                                                               \
  template Status fnName(GPU::Surface&, int64_t, int64_t, float, float, float, float, uint32_t, GPU::Stream);  \
  template Status fnName(GPU::Buffer<uint32_t>&, int64_t, int64_t, float, float, float, float, uint32_t, GPU::Stream);

CIRCLE_FN(drawCircle, circle)
CIRCLE_FN(drawCircleTop, circleTop)
CIRCLE_FN(drawCircleBottom, circleBottom)
CIRCLE_FN(drawCircleTopRight, circleTopRight)
CIRCLE_FN(drawCircleBottomRight, circleBottomRight)

}  // namespace Render
}  // namespace VideoStitch
