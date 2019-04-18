// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "2dBuffer.hpp"
#include "buffer.hpp"
#include "surface.hpp"
#include "hostBuffer.hpp"
#include "stream.hpp"

namespace VideoStitch {
namespace GPU {

// Device --> Device
// * async
template <typename T>
Status memcpyAsync(Buffer<T> dst, Buffer<const T> src, size_t copySize, const Stream& stream);

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, Buffer<T> src, size_t copySize, const Stream& stream) {
  return memcpyAsync(dst, src.as_const(), copySize, stream);
}

// * blocking
template <typename T>
Status memcpyBlocking(Buffer<T> dst, Buffer<const T> src, size_t copySize);

template <typename T>
inline Status memcpyBlocking(Buffer<T> dst, Buffer<T> src, size_t copySize) {
  return memcpyBlocking(dst, src.as_const(), copySize);
}

template <typename T>
inline Status memcpyBlocking(Buffer<T> dst, Buffer<T> src) {
  return memcpyBlocking(dst, src.as_const());
}

template <typename T>
inline Status memcpyBlocking(Buffer<T> dst, Buffer<const T> src) {
  if (dst.byteSize() < src.byteSize()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyBlocking(dst, src, src.byteSize());
}

// Host --> Device
// * async
template <typename T>
Status memcpyAsync(Buffer<T> dst, const T* src, size_t copySize, const Stream& stream);

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, const T* src, const Stream& stream) {
  return memcpyAsync(dst, src, dst.byteSize(), stream);
}

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, HostBuffer<const T> src, const Stream& stream) {
  if (dst.byteSize() < src.byteSize()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyAsync(dst, src.hostPtr(), src.byteSize(), stream);
}

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, HostBuffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= src.byteSize());
  return memcpyAsync(dst, src.hostPtr(), copySize, stream);
}

template <typename T>
inline Status memcpyAsync(Buffer<T> dst, HostBuffer<T> src, size_t copySize, const Stream& stream) {
  return memcpyAsync(dst, src.as_const(), copySize, stream);
}

// * blocking
template <typename T>
Status memcpyBlocking(Buffer<T> dst, const T* src, size_t copySize);

template <typename T>
inline Status memcpyBlocking(Buffer<T> dst, const T* src) {
  return memcpyBlocking(dst, src, dst.byteSize());
}

template <typename T>
inline Status memcpyBlocking(typename Buffer<T>::PotentialBuffer dst, const T* src) {
  if (!dst.status()) {
    return dst.status();
  }
  return memcpyBlocking(dst.value(), src, dst.byteSize());
}

// Copy memory Device --> Host
// * async
template <typename T>
Status memcpyAsync(T* dst, Buffer<const T> src, size_t copySize, const Stream& stream);

template <typename T>
inline Status memcpyAsync(T* dst, Buffer<const T> src, const Stream& stream) {
  return memcpyAsync(dst, src, src.byteSize(), stream);
}

template <typename T>
inline Status memcpyAsync(HostBuffer<T> dst, Buffer<const T> src, size_t copySize, const Stream& stream) {
  assert(copySize <= dst.byteSize());
  return memcpyAsync(dst.hostPtr(), src, copySize, stream);
}

template <typename T>
inline Status memcpyAsync(HostBuffer<T> dst, Buffer<const T> src, const Stream& stream) {
  if (dst.byteSize() < src.byteSize()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyAsync(dst.hostPtr(), src, src.byteSize(), stream);
}

Status memcpyAsync(unsigned char* dst, Buffer2D src, const Stream& stream);
inline Status memcpyAsync(HostBuffer<unsigned char> dst, Buffer2D src, const Stream& stream) {
  if (dst.byteSize() < src.getWidth() * src.getHeight()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyAsync(dst.hostPtr(), src, stream);
}

// * blocking
template <typename T>
Status memcpyBlocking(T* dst, Buffer<const T> src, size_t copySize);

template <typename T>
inline Status memcpyBlocking(T* dst, Buffer<const T> src) {
  return memcpyBlocking(dst, src, src.byteSize());
}

template <typename T>
inline Status memcpyBlocking(T* dst, Buffer<T> src) {
  return memcpyBlocking(dst, src.as_const());
}

template <typename T>
inline Status memcpyBlocking(HostBuffer<T> dst, Buffer<const T> src) {
  if (dst.byteSize() < src.byteSize()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyBlocking(dst.hostPtr(), src);
}

Status memcpyBlocking(unsigned char* dst, Buffer2D src);
inline Status memcpyBlocking(HostBuffer<unsigned char> dst, Buffer2D src) {
  if (dst.byteSize() < src.getWidth() * src.getHeight()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyBlocking(dst.hostPtr(), src);
}

Status memcpyBlocking(Buffer2D dst, const unsigned char* src);
inline Status memcpyBlocking(Buffer2D dst, HostBuffer<unsigned char> src) {
  if (src.byteSize() < dst.getWidth() * dst.getHeight()) {
    return {Origin::GPU, ErrType::ImplementationError, "Copy destination is too small"};
  }
  return memcpyBlocking(dst, src.hostPtr());
}

// Memset
// Note: memset to value implemented in render.hpp

// Set settingSize Bytes of GPU memory to 0
template <typename T>
Status memsetToZeroAsync(Buffer<T> devPtr, size_t settingSize, const Stream& stream);

template <typename T>
inline Status memsetToZeroAsync(Buffer<T> devPtr, const Stream& stream) {
  return memsetToZeroAsync(devPtr, devPtr.byteSize(), stream);
}

template <typename T>
Status memsetToZeroBlocking(Buffer<T> devPtr, size_t settingSize);

Status memsetToZeroAsync(Surface& dst, const Stream& stream);

// Device buffer to texture memory
template <typename T>
Status memcpyAsync(Surface& dst, Buffer<const T> src, const Stream& stream);
Status memcpyBlocking(Surface& dst, Buffer<const uint32_t> src);

// Texture memory to device buffer
template <typename T>
Status memcpyAsync(Buffer<T> dst, Surface& src, const Stream& stream);
template <typename T>
Status memcpyBlocking(Buffer<T> dst, Surface& src);

// Texture memory to host buffer
template <typename T>
Status memcpyAsync(T* dst, Surface& src, const Stream& stream);

template <typename T>
Status memcpyBlocking(T* dst, Surface& src);

// Host to texture memory
Status memcpyAsync(Surface& dst, uint32_t* src, const Stream& stream);
Status memcpyBlocking(Surface& dst, uint32_t* src);

Status memcpy2DAsync(Buffer<uint32_t> dst, Buffer<uint32_t> src, size_t src_origin_width, size_t src_origin_height,
                     size_t dst_origin_width, size_t dst_origin_height, size_t region_width, size_t region_height,
                     size_t src_pitch, size_t dst_pitch, const Stream& stream);

Status memcpyCubemapAsync(CubemapSurface& dst, Buffer<uint32_t> srcXP, Buffer<uint32_t> srcXN, Buffer<uint32_t> srcYP,
                          Buffer<uint32_t> srcYN, Buffer<uint32_t> srcZP, Buffer<uint32_t> srcZN, size_t faceDim,
                          const Stream& stream);

}  // namespace GPU
}  // namespace VideoStitch
