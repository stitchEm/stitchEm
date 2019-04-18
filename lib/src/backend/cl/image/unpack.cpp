// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <image/unpack.hpp>

#include "../deviceBuffer2D.hpp"
#include "../context.hpp"
#include "../kernel.hpp"
#include "../surface.hpp"

#include "gpu/memcpy.hpp"
#include "gpu/util.hpp"

#ifdef VS_OPENCL

namespace VideoStitch {
namespace Image {

namespace {
#include "unpack.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(unpack, true);

Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t /*width*/,
                  std::size_t /*height*/, GPU::Stream stream) {
  size_t src_origin[3] = {0, 0, 0};
  size_t dst_origin[3] = {0, 0, 0};
  size_t region[3] = {dst.getWidth(), dst.getHeight(), 1};
  return CL_ERROR(clEnqueueCopyBufferRect(stream.get(), src.get(), dst.get(), src_origin, dst_origin, region, 0,
                                          0,                  // src_row_pitch, src_slice_pitch
                                          dst.getPitch(), 0,  // dst_row_pitch, dst_slice_pitch
                                          0, 0, nullptr));
}

Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                  GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelRGBA))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), (unsigned)dst.getPitch(), src.get(), (unsigned)(width),
                                        (unsigned)(height));
}

Status unpackRGB(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                 GPU::Stream stream) {
  auto kernel2D =
      GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelRGB)).setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), (unsigned)dst.getPitch(), src.get().raw(), (unsigned)(width),
                                        (unsigned)(height));
}

Status unpackRGB(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                 GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelRGBSource))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), (unsigned)dst.getPitch(), src.get(), (unsigned)(width),
                                        (unsigned)(height));
}

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t /* width */,
                   std::size_t /* height */, GPU::Stream stream) {
  size_t src_origin[3] = {0, 0, 0};
  size_t dst_origin[3] = {0, 0, 0};
  size_t region[3] = {dst.getWidth(), dst.getHeight(), 1};
  return CL_ERROR(clEnqueueCopyBufferRect(stream.get(), src.get(), dst.get(), src_origin, dst_origin, region, 0,
                                          0,                  // src_row_pitch, src_slice_pitch
                                          dst.getPitch(), 0,  // dst_row_pitch, dst_slice_pitch
                                          0, 0, nullptr));
}

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                   GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelF32C1))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), (unsigned)dst.getPitch(), src.get(), (unsigned)(width),
                                        (unsigned)(height));
}

Status unpackGrayscale16(GPU::Buffer2D& /* dst */, const GPU::Buffer<const uint32_t>& /* input */, size_t /* width */,
                         size_t /* height */, GPU::Stream /* s */) {
  // TODO_OPENCL_IMPL/
  return {Origin::GPU, ErrType::UnsupportedAction,
          "Color space conversion for Grayscale16 not implemented from buffer"};
}

Status unpackGrayscale16(GPU::Buffer2D& /* dst */, const GPU::Surface& /* surf */, size_t /* width */,
                         size_t /* height */, GPU::Stream /* s */) {
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for Grayscale16 not implemented in OpenCL"};
}

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                   const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelDepth))
                      .setup2D(stream, (unsigned)((width + 1) / 2), (unsigned)((height + 1) / 2));
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(),
                                        (unsigned)uDst.getPitch(), vDst.get().raw(), (unsigned)vDst.getPitch(),
                                        (float*)src.get().raw(), (unsigned)width, (unsigned)height);
}

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& src,
                   std::size_t width, std::size_t height, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelDepthSource))
                      .setup2D(stream, (unsigned)((width + 1) / 2), (unsigned)((height + 1) / 2));
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(),
                                        (unsigned)uDst.getPitch(), vDst.get().raw(), (unsigned)vDst.getPitch(),
                                        src.get(), (unsigned)width, (unsigned)height);
}

Status unpackYV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                  const GPU::Buffer<const uint32_t>& array, size_t width, size_t height, GPU::Stream stream) {
  // planar colorspace, 3 planes
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelYV12))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(),
                                        (unsigned)uDst.getPitch(), vDst.get().raw(), (unsigned)vDst.getPitch(),
                                        array.get().raw(), (unsigned)(width), (unsigned)(height));
}

Status unpackYV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& array,
                  size_t width, size_t height, GPU::Stream stream) {
  // planar colorspace, 3 planes
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelYV12Source))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(),
                                        (unsigned)uDst.getPitch(), vDst.get().raw(), (unsigned)vDst.getPitch(),
                                        array.get(), (unsigned)(width), (unsigned)(height));
}

Status unpackNV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uvDst, const GPU::Buffer<const uint32_t>& array, size_t width,
                  size_t height, GPU::Stream stream) {
  // planar colorspace, 2 planes
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelNV12))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uvDst.get().raw(),
                                        (unsigned)uvDst.getPitch(), array.get().raw(), (unsigned)(width),
                                        (unsigned)(height));
}

Status unpackNV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uvDst, const GPU::Surface& array, size_t width, size_t height,
                  GPU::Stream stream) {
  // planar colorspace, 2 planes
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelNV12Source))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch(), uvDst.get().raw(),
                                        (unsigned)uvDst.getPitch(), array.get(), (unsigned)(width), (unsigned)(height));
}

Status unpackYUY2(GPU::Buffer2D&, const GPU::Buffer<const uint32_t>&, std::size_t, std::size_t, GPU::Stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for YUV422 not implemented"};
}

Status unpackYUY2(GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t, GPU::Stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for YUV422 not implemented"};
}

Status unpackUYVY(GPU::Buffer2D&, const GPU::Buffer<const uint32_t>&, std::size_t, std::size_t, GPU::Stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for YUV422 not implemented"};
}

Status unpackUYVY(GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t, GPU::Stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for YUV422 not implemented"};
}

Status unpackYUV422P10(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                       const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                       GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackYUV422P10Kernel))
                      .setup2D(stream, (unsigned)((width + 1) / 2), (unsigned)height);

  return kernel2D.enqueueWithKernelArgs(yDst.get().raw(), (unsigned)yDst.getPitch() / 2, uDst.get().raw(),
                                        (unsigned)uDst.getPitch() / 2, vDst.get().raw(), (unsigned)vDst.getPitch() / 2,
                                        src.get(), (unsigned)width, (unsigned)height);
}

Status unpackYUV422P10(GPU::Buffer2D&, GPU::Buffer2D&, GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t,
                       GPU::Stream) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for YUV422P10 not implemented"};
}

Status unpackGrayscale(GPU::Buffer2D& /*dst*/, const GPU::Buffer<const uint32_t>& /*src*/, std::size_t /*width*/,
                       std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for Grayscale not implemented"};
}

Status unpackGrayscale(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                       GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(unpackKernelGrayscaleSource))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), (unsigned)dst.getPitch(), src.get(), (unsigned)(width),
                                        (unsigned)(height));
}

// --------------------------------------------------------------------------

Status convertRGBToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                        GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertRGBToRGBAKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src, (unsigned)width, (unsigned)height);
}

Status convertRGB210ToRGBA(GPU::Surface& dst, GPU::Buffer<const uint32_t> src, std::size_t width, std::size_t height,
                           GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertRGB210ToRGBAKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src, (unsigned)width, (unsigned)height);
}

Status unpackYUV422P10(GPU::Buffer<unsigned char> dst, GPU::Buffer<const uint32_t> src, std::size_t width,
                       std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  std::string kernelName = KERNEL_STR(unpackYUV422P10Kernel);
  auto kernel1D = GPU::Kernel::get(PROGRAM(unpack), kernelName).setup1D(stream, (unsigned)(width * height / 2));
  return kernel1D.enqueueWithKernelArgs(dst.as<unsigned short>(), src, (unsigned)width, (unsigned)height);
}

Status unpackGrayscale(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                       std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion for Grayscale not implemented"};
  // const dim3 dimBlock2D(16, 16, 1);
  // const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y),
  // 1); unpackKernelGrayscale <<< dimGrid2D, dimBlock2D, 0, stream.get() >>>(dst, src, (unsigned)width,
  // (unsigned)height);
}

Status convertBGRToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                        std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion BGR to RGBA not implemented"};
}

Status convertBGRUToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                         std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion BGRU to RGBA not implemented"};
}

Status convertBayerRGGBToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                              std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion RGGB to RGBA not implemented"};
}

Status convertBayerBGGRToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                              std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion BayerBGGR to RGBA not implemented"};
}

Status convertBayerGRBGToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                              std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion BayerGRBG to RGBA not implemented"};
}

Status convertBayerGBRGToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                              std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion BayerGBRG to RGBA not implemented"};
}

Status convertUYVYToRGBA(GPU::Surface& /*dst*/, GPU::Buffer<const unsigned char> /*src*/, std::size_t /*width*/,
                         std::size_t /*height*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::GPU, ErrType::UnsupportedAction, "Color space conversion UYVY to RGBA not implemented"};
}

Status convertYUV422P10ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertYUV422P10ToRGBAKernel))
                      .setup2D(stream, (unsigned)width / 2, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src.as<const uint16_t>().get(), (unsigned)(width),
                                        (unsigned)(height));
}

Status convertYUY2ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  std::string kernelName = KERNEL_STR(convertYUY2ToRGBAKernel);
  auto kernel1D = GPU::Kernel::get(PROGRAM(unpack), kernelName).setup1D(stream, (unsigned)(width * height / 2));
  return kernel1D.enqueueWithKernelArgs(dst, src, (unsigned)width, (unsigned)height);
}

Status convertYV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertYV12ToRGBAKernel))
                      .setup2D(stream, (unsigned)width / 2, (unsigned)height / 2);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src, (unsigned)width, (unsigned)height);
}

Status convertNV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertNV12ToRGBAKernel))
                      .setup2D(stream, (unsigned)width / 2, (unsigned)height / 2);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src, (unsigned)width, (unsigned)height);
}

Status convertGrayscaleToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(unpack), KERNEL_STR(convertGrayscaleToRGBAKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst.get().raw(), src, (unsigned)width, (unsigned)height);
}

}  // namespace Image
}  // namespace VideoStitch

#endif  // VS_OPENCL
