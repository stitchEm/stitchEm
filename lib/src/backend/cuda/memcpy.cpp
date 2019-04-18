// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/gpu_device.hpp"
#include "gpu/memcpy.hpp"

#include "deviceBuffer.hpp"
#include "deviceBuffer2D.hpp"
#include "surface.hpp"
#include "deviceStream.hpp"

#include "cuda/error.hpp"

#include <cuda_runtime.h>

namespace VideoStitch {
namespace GPU {

// Device --> Device
// * async
template <typename T>
Status memcpyAsync(Buffer<T> dst, Buffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= src.byteSize() && copySize <= dst.byteSize());
  return CUDA_ERROR(cudaMemcpyAsync(dst.get(), src.get(), copySize, cudaMemcpyDeviceToDevice, stream.get()));
}

// * blocking
template <typename T>
Status memcpyBlocking(GPU::Buffer<T> dst, GPU::Buffer<const T> src, size_t copySize) {
  assert(copySize <= src.byteSize() && copySize <= dst.byteSize());
  FAIL_RETURN(CUDA_ERROR(cudaDeviceSynchronize()));
  return CUDA_ERROR(cudaMemcpy(dst.get(), src.get(), copySize, cudaMemcpyDeviceToDevice));
}

// Host --> Device
// * async
template <typename T>
Status memcpyAsync(Buffer<T> dst, const T* src, size_t copySize, const Stream& stream) {
  assert(copySize <= dst.byteSize());
  return CUDA_ERROR(cudaMemcpyAsync(dst.get(), src, copySize, cudaMemcpyHostToDevice, stream.get()));
}

// * blocking
template <typename T>
Status memcpyBlocking(GPU::Buffer<T> dst, const T* src, size_t copySize) {
  assert(copySize <= dst.byteSize());
  FAIL_RETURN(CUDA_ERROR(cudaDeviceSynchronize()));
  FAIL_RETURN(CUDA_ERROR(cudaMemcpy(dst.get(), src, copySize, cudaMemcpyHostToDevice)));
  return CUDA_ERROR(cudaDeviceSynchronize());
}

// Device --> Host
// * async
template <typename T>
Status memcpyAsync(T* dst, GPU::Buffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= src.byteSize());
  return CUDA_ERROR(cudaMemcpyAsync(dst, src.get(), copySize, cudaMemcpyDeviceToHost, stream.get()));
}

Status memcpyAsync(unsigned char* dst, Buffer2D src, const Stream& stream) {
  if ((src.getWidth() == 0) || (src.getHeight() == 0)) {
    return Status::OK();
  }
  return CUDA_ERROR(cudaMemcpy2DAsync((void*)dst, src.getWidth(), src.get().raw(), src.getPitch(), src.getWidth(),
                                      src.getHeight(), cudaMemcpyDeviceToHost, stream.get()));
}

// * blocking
template <typename T>
Status memcpyBlocking(T* dst, GPU::Buffer<const T> src, size_t copySize) {
  assert(copySize <= src.byteSize());
  FAIL_RETURN(CUDA_ERROR(cudaDeviceSynchronize()));
  return CUDA_ERROR(cudaMemcpy(dst, src.get(), copySize, cudaMemcpyDeviceToHost));
}

Status memcpyBlocking(unsigned char* dst, Buffer2D src) {
  if ((src.getWidth() == 0) || (src.getHeight() == 0)) {
    return Status::OK();
  }
  return CUDA_ERROR(cudaMemcpy2D((void*)dst, src.getWidth(), src.get().raw(), src.getPitch(), src.getWidth(),
                                 src.getHeight(), cudaMemcpyDeviceToHost));
}

Status memcpyBlocking(Buffer2D dst, const unsigned char* src) {
  if ((dst.getWidth() == 0) || (dst.getHeight() == 0)) {
    return Status::OK();
  }
  return CUDA_ERROR(cudaMemcpy2D((void*)dst.get().raw(), dst.getPitch(), src, dst.getWidth(), dst.getWidth(),
                                 dst.getHeight(), cudaMemcpyHostToDevice));
}

Status memsetToZeroAsync(void* devPtr, size_t count, const Stream& stream) {
  return CUDA_ERROR(cudaMemsetAsync(devPtr, 0, count, stream.get()));
}

template <typename T>
Status memsetToZeroAsync(GPU::Buffer<T> devPtr, size_t count, const Stream& stream) {
  assert(count <= devPtr.byteSize());
  return memsetToZeroAsync(devPtr.get(), count, stream);
}

Status memsetToZeroBlocking(void* devPtr, size_t count) {
  FAIL_RETURN(CUDA_ERROR(cudaDeviceSynchronize()));
  FAIL_RETURN(CUDA_ERROR(cudaMemsetAsync(devPtr, 0, count)));
  FAIL_RETURN(CUDA_ERROR(cudaDeviceSynchronize()));
  return CUDA_STATUS;
}

template <typename T>
Status memsetToZeroBlocking(GPU::Buffer<T> devPtr, size_t count) {
  assert(count <= devPtr.byteSize());
  return memsetToZeroBlocking(devPtr.get(), count);
}

Status memcpyBlocking(GPU::Surface& dst, GPU::Buffer<const uint32_t> src) {
  return CUDA_ERROR(cudaMemcpyToArray(dst.get().array_, 0, 0, src.get(), dst.width() * dst.height() * sizeof(uint32_t),
                                      cudaMemcpyDeviceToDevice));
}

template <typename T>
Status memcpyBlocking(GPU::Buffer<T> dst, GPU::Surface& src) {
  return CUDA_ERROR(cudaMemcpyFromArray((void*)dst.get().raw(), src.get().array(), 0, 0,
                                        src.width() * src.height() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template Status memcpyBlocking(Buffer<uint32_t> dst, Surface& src);
template Status memcpyBlocking(Buffer<float2> dst, Surface& src);

template <typename T>
Status memcpyBlocking(T* dst, GPU::Surface& src) {
  return CUDA_ERROR(cudaMemcpyFromArray((void*)dst, src.get().array_, 0, 0, src.width() * src.height() * sizeof(T),
                                        cudaMemcpyDeviceToHost));
}

Status memcpyBlocking(GPU::Surface& dst, uint32_t* src) {
  return CUDA_ERROR(cudaMemcpyToArray(dst.get().array_, 0, 0, src, dst.width() * dst.height() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice));
}

// TODO_OPENCL_IMPL template T should be something similar to PixelType, capturing channel order and data type
template <typename T>
Status memcpyAsync(GPU::Surface& dst, GPU::Buffer<const T> src, const Stream& stream) {
  return CUDA_ERROR(cudaMemcpyToArrayAsync(dst.get().array_, 0, 0, src.get(), dst.width() * dst.height() * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream.get()));
}
template Status memcpyAsync(GPU::Surface& dst, GPU::Buffer<const uint32_t> src, const Stream& stream);
template Status memcpyAsync(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, const Stream& stream);

template <typename T>
Status memcpyAsync(GPU::Buffer<T> dst, GPU::Surface& src, const Stream& stream) {
  return CUDA_ERROR(cudaMemcpyFromArrayAsync((void*)dst.get().raw(), src.get().array_, 0, 0,
                                             src.width() * src.height() * sizeof(T), cudaMemcpyDeviceToDevice,
                                             stream.get()));
}
template Status memcpyAsync(GPU::Buffer<uint32_t> dst, GPU::Surface& src, const Stream& stream);
template Status memcpyAsync(GPU::Buffer<unsigned char> dst, GPU::Surface& src, const Stream& stream);

template <typename T>
Status memcpyAsync(T* dst, GPU::Surface& src, const Stream& stream) {
  return CUDA_ERROR(cudaMemcpyFromArrayAsync((void*)dst, src.get().array_, 0, 0,
                                             src.width() * src.height() * sizeof(float), cudaMemcpyDeviceToHost,
                                             stream.get()));
}

Status memcpyAsync(GPU::Surface& dst, uint32_t* src, const Stream& stream) {
  return CUDA_ERROR(cudaMemcpyToArrayAsync(dst.get().array_, 0, 0, src, dst.width() * dst.height() * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice, stream.get()));
}

Status memcpy2DAsync(Buffer<uint32_t> dst, Buffer<uint32_t> src, size_t src_origin_width, size_t src_origin_height,
                     size_t dst_origin_width, size_t dst_origin_height, size_t region_width, size_t region_height,
                     size_t src_pitch, size_t dst_pitch, const Stream& stream) {
  uint32_t* dst_ptr = dst.get();
  uint32_t* src_ptr = src.get();
  dst_ptr += dst_pitch * dst_origin_height + dst_origin_width;
  src_ptr += src_pitch * src_origin_height + src_origin_width;
  return CUDA_ERROR(cudaMemcpy2DAsync(dst_ptr, dst_pitch * sizeof(uint32_t), src_ptr, src_pitch * sizeof(uint32_t),
                                      region_width * sizeof(uint32_t), region_height, cudaMemcpyDeviceToDevice,
                                      stream.get()));
}

// Template instantiations, shared between backends
#include "../common/memcpy.inst"

}  // namespace GPU
}  // namespace VideoStitch
