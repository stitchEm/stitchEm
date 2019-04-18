// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifdef VS_OPENCL
#error This file is part of the CUDA backend. It is not supposed to be included in libvideostitch-OpenCL.
#endif

#include "gpu/buffer.hpp"

namespace VideoStitch {
namespace GPU {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : data(nullptr) {}

  explicit DeviceBuffer(T *buf) : data(buf) {}

  static Buffer<T> createBuffer(T *buf, size_t elems);

  T *raw() const { return data; }

  operator T *() { return data; }
  operator T *() const { return data; }

  bool operator==(const DeviceBuffer &other) const { return data == other.data; }

  template <typename S>
  friend class Buffer;
  friend class Buffer2D;

 private:
  T *data;
};

}  // namespace GPU
}  // namespace VideoStitch
