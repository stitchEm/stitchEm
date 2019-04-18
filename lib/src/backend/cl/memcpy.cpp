// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/gpu_device.hpp"
#include "gpu/memcpy.hpp"

#include "gpu/vectorTypes.hpp"

#include "deviceBuffer2D.hpp"
#include "deviceBuffer.hpp"
#include "deviceHostBuffer.hpp"
#include "deviceStream.hpp"
#include "surface.hpp"

#ifdef __APPLE__
#include "kernel.hpp"
#endif  // __APPLE__

#include "cl_error.hpp"
#include "opencl.h"

namespace VideoStitch {
namespace GPU {

#ifdef __APPLE__

namespace {

#include "memset.xxd"
}
INDIRECT_REGISTER_OPENCL_PROGRAM(memset, true);

#endif  // __APPLE__

template <typename Functor>
Status blockingOperation(Functor asyncOperation) {
  const auto& stream = Stream::getDefault();
  PROPAGATE_FAILURE_STATUS(asyncOperation(stream));
  return stream.synchronize();
}

// Device --> Device
// * async
template <typename T>
Status memcpyAsync(Buffer<T> dst, Buffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= dst.byteSize());
  assert(copySize <= src.byteSize());
  return CL_ERROR(clEnqueueCopyBuffer(stream.get(), src.get(), dst.get(), 0, 0, copySize, 0, nullptr, nullptr));
}

// * blocking
template <typename T>
Status memcpyBlocking(Buffer<T> dst, Buffer<const T> src, size_t copySize) {
  auto copy = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, copySize, stream); };
  return blockingOperation(copy);
}

// Host --> Device
// * async

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, const T* src, size_t copySize, const Stream& stream, bool blockingWrite) {
  assert(copySize <= dst.byteSize());
  return CL_ERROR(clEnqueueWriteBuffer(stream.get(), dst.get(), blockingWrite ? CL_TRUE : CL_FALSE, 0, copySize, src, 0,
                                       nullptr, nullptr));
}

template <typename T>
Status memcpyAsync(Buffer<T> dst, const T* src, size_t copySize, const Stream& stream) {
  return memcpyAsync(dst, src, copySize, stream, false);
}

// * blocking
template <typename T>
Status memcpyBlocking(Buffer<T> dst, const T* src, size_t copySize) {
  auto copy = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, copySize, stream, true); };
  return blockingOperation(copy);
}

// Device --> Host
// * async
template <typename T>
Status memcpyAsync(T* dst, Buffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= src.byteSize());
  return CL_ERROR(clEnqueueReadBuffer(stream.get(), src.get(), CL_FALSE, 0, copySize, dst, 0, nullptr, nullptr));
}

Status memcpyAsync(unsigned char* dst, Buffer2D src, const Stream& stream) {
  size_t buffer_origin[3] = {0, 0, 0};
  size_t host_origin[3] = {0, 0, 0};
  size_t region[3] = {src.getWidth(), src.getHeight(), 1};
  return CL_ERROR(clEnqueueReadBufferRect(stream.get(), src.get().raw(),
                                          CL_FALSE,  // blocking_read,
                                          buffer_origin, host_origin, region, src.getPitch(),
                                          0,  // buffer_slice_pitch,
                                          0,  // host_row_pitch,
                                          0,  // host_slice_pitch,
                                          (void*)dst, 0, 0, nullptr));
}

Status memcpyAsync(Buffer2D dst, const unsigned char* src, const Stream& stream) {
  size_t buffer_origin[3] = {0, 0, 0};
  size_t host_origin[3] = {0, 0, 0};
  size_t region[3] = {dst.getWidth(), dst.getHeight(), 1};
  return CL_ERROR(clEnqueueWriteBufferRect(stream.get(), dst.get().raw(),
                                           CL_FALSE,  // blocking_write,
                                           buffer_origin, host_origin, region, dst.getPitch(),
                                           0,  // buffer_slice_pitch,
                                           0,  // host_row_pitch,
                                           0,  // host_slice_pitch,
                                           (void*)src, 0, 0, nullptr));
}

// * blocking
template <typename T>
Status memcpyBlocking(T* dst, Buffer<const T> src, size_t copySize) {
  auto copy = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, copySize, stream); };
  return blockingOperation(copy);
}

Status memcpyBlocking(unsigned char* dst, Buffer2D src) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, stream); };
  return blockingOperation(asyncSet);
}

Status memcpyBlocking(Buffer2D dst, const unsigned char* src) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, stream); };
  return blockingOperation(asyncSet);
}

#ifdef __APPLE__
template <typename T>
Status memsetToZeroAsync(Buffer<T> devPtr, size_t copySize, const Stream& stream) {
  auto kernel = Kernel::get(PROGRAM(memset), KERNEL_STR(memsetToZero)).setup1D(stream, (unsigned)copySize);
  return kernel.enqueueWithKernelArgs(devPtr.get(), (unsigned)copySize);
}
#else   // __APPLE__
template <typename T>
Status memsetToZeroAsync(Buffer<T> devPtr, size_t copySize, const Stream& stream) {
  T zero = {0};
  return CL_ERROR(clEnqueueFillBuffer(stream.get(), devPtr.get(), &zero, sizeof(T), 0, copySize, 0, nullptr, nullptr));
}
#endif  // __APPLE__

Status memsetToZeroAsync(Surface& dst, const Stream& stream) {
  uint32_t zero[4] = {0, 0, 0, 0};
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {dst.width(), dst.height(), 1};
  return CL_ERROR(clEnqueueFillImage(stream.get(), dst.get(), &zero, origin, region, 0, nullptr, nullptr));
}

template <typename T>
Status memsetToZeroBlocking(Buffer<T> devPtr, size_t count) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memsetToZeroAsync(devPtr, count, stream); };
  return blockingOperation(asyncSet);
}

template <typename T>
Status memcpyAsync(Surface& dst, Buffer<const T> src, const Stream& stream) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {dst.width(), dst.height(), 1};
  return CL_ERROR(
      clEnqueueCopyBufferToImage(stream.get(), src.get(), dst.get(), 0, origin, region, 0, nullptr, nullptr));
}

template Status memcpyAsync(Surface& dst, Buffer<const uint32_t> src, const Stream& stream);
template Status memcpyAsync(Surface& dst, Buffer<const unsigned char> src, const Stream& stream);

Status memcpyBlocking(Surface& dst, Buffer<const uint32_t> src) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, stream); };
  return blockingOperation(asyncSet);
}

template <typename T>
Status memcpyAsync(Buffer<T> dst, Surface& src, const Stream& stream) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {src.width(), src.height(), 1};
  return CL_ERROR(
      clEnqueueCopyImageToBuffer(stream.get(), src.get(), dst.get(), origin, region, 0, 0, nullptr, nullptr));
}

template Status memcpyAsync(Buffer<uint32_t> dst, Surface& src, const Stream& stream);
template Status memcpyAsync(Buffer<unsigned char> dst, Surface& src, const Stream& stream);
template Status memcpyAsync(Buffer<float2> dst, Surface& src, const Stream& stream);

template <typename T>
Status memcpyBlocking(Buffer<T> dst, Surface& src) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, stream); };
  return blockingOperation(asyncSet);
}

template Status memcpyBlocking(Buffer<float2> dst, Surface& src);

template <typename T>
Status memcpyAsync(T* dst, Surface& src, const Stream& stream) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {src.width(), src.height(), 1};
  return CL_ERROR(
      clEnqueueReadImage(stream.get(), src.get(), CL_FALSE, origin, region, 0, 0, dst, 0, nullptr, nullptr));
}

Status memcpyAsync(Surface& dst, uint32_t* src, const Stream& stream) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {dst.width(), dst.height(), 1};
  return CL_ERROR(
      clEnqueueWriteImage(stream.get(), dst.get(), CL_FALSE, origin, region, 0, 0, src, 0, nullptr, nullptr));
}

template <typename T>
Status memcpyBlocking(T* dst, Surface& src) {
  auto asyncSet = [&](const Stream& stream) -> Status { return memcpyAsync(dst, src, stream); };
  return blockingOperation(asyncSet);
}

Status memcpy2DAsync(Buffer<uint32_t> dst, Buffer<uint32_t> src, size_t src_origin_width, size_t src_origin_height,
                     size_t dst_origin_width, size_t dst_origin_height, size_t region_width, size_t region_height,
                     size_t src_pitch, size_t dst_pitch, const Stream& stream) {
  size_t src_origin[3] = {src_origin_width * sizeof(uint32_t), src_origin_height * sizeof(uint32_t), 0};
  size_t dst_origin[3] = {dst_origin_width * sizeof(uint32_t), dst_origin_height * sizeof(uint32_t), 0};
  size_t region[3] = {region_width * sizeof(uint32_t), region_height * sizeof(uint32_t), 1};
  return CL_ERROR(clEnqueueCopyBufferRect(stream.get(), src.get(), dst.get(), src_origin, dst_origin, region,
                                          src_pitch * sizeof(uint32_t), 0, dst_pitch * sizeof(uint32_t), 0, 0, nullptr,
                                          nullptr));
}

Status memcpyCubemapAsync(CubemapSurface& /*cubemapSurface*/, Buffer<uint32_t> /*xPosPbo*/,
                          Buffer<uint32_t> /*xNegPbo*/, Buffer<uint32_t> /*yPosPbo*/, Buffer<uint32_t> /*yNegPbo*/,
                          Buffer<uint32_t> /*zPosPbo*/, Buffer<uint32_t> /*zNegPbo*/, size_t /*faceDim*/,
                          const Stream& /*stream*/) {
  assert(false);
  return Status::OK();
}

// Template instantiations, shared between backends
#include "../common/memcpy.inst"

}  // namespace GPU
}  // namespace VideoStitch
