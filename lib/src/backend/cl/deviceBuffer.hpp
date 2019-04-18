// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/buffer.hpp>

#include "opencl.h"

namespace VideoStitch {
namespace GPU {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : data(nullptr) {}

  explicit DeviceBuffer(const DeviceBuffer *other) : data(other ? other->data : nullptr) {}

  template <typename S>
  explicit DeviceBuffer(const DeviceBuffer<S> *other) : data(other ? other->data : nullptr) {}

  explicit DeviceBuffer(cl_mem buf) : data(buf) {}

  static Buffer<T> createBuffer(cl_mem buf, size_t elems);

  cl_mem raw() const { return data; }

  operator cl_mem() const { return raw(); }

  bool operator==(const DeviceBuffer &other) const { return data == other.data; }

  friend class Buffer<T>;

  template <typename S>
  friend class DeviceBuffer;

 private:
  cl_mem data;
};

}  // namespace GPU
}  // namespace VideoStitch
