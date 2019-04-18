// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <vector>

namespace VideoStitch {
namespace GPU {

// Backend specific pimpl class
class DeviceSurface;
class DeviceCubemapSurface;
class Stream;

/** A chunk of 2D memory on a GPU device,
 *  with access to the hardware cache capabilities (texture cache).
 */
class Surface {
 public:
  size_t width() const { return _width; }

  size_t height() const { return _height; }

  ~Surface();

  void swap(Surface& other) {
    std::swap(_width, other._width);
    std::swap(_height, other._height);
    std::swap(pimpl, other.pimpl);
  }

  Surface& operator=(Surface& other) {
    swap(other);
    return *this;
  }

  bool operator==(const Surface& other) const;

  bool operator!=(const Surface& other) const { return !(*this == other); }

  /**
   * Print name and size of currently allocated Buffers.
   */
  static void printPoolStats();

  Surface(DeviceSurface* pimpl, size_t width, size_t height);

 private:
  Surface(const Surface& other);

  DeviceSurface* pimpl;

  size_t _width;
  size_t _height;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  DeviceSurface& get();
  const DeviceSurface& get() const;
};

/** A chunk of cubemap memory on a GPU device,
 *  with access to the hardware cache capabilities (texture cache).
 */
class CubemapSurface {
 public:
  size_t length() const { return _length; }

  ~CubemapSurface();

  void swap(CubemapSurface& other) {
    std::swap(_length, other._length);
    std::swap(pimpl, other.pimpl);
  }

  CubemapSurface& operator=(CubemapSurface& other) {
    swap(other);
    return *this;
  }

  bool operator==(const CubemapSurface& other) const;

  bool operator!=(const CubemapSurface& other) const { return !(*this == other); }

  /**
   * Print name and size of currently allocated Buffers.
   */
  static void printPoolStats();

  CubemapSurface(DeviceCubemapSurface* pimpl, size_t length);

 private:
  CubemapSurface(const CubemapSurface& other) = delete;

  DeviceCubemapSurface* pimpl;

  size_t _length;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  DeviceCubemapSurface& get();
  const DeviceCubemapSurface& get() const;
};

/**
 * Print name and size of currently allocated Cached Buffers.
 */
void printSurfacePoolStats();

/**
 * Total size in Bytes of currently allocated Buffers.
 */
std::size_t getSurfacePoolCurrentSize();

/**
 * Total size in Bytes of currently allocated Buffers, by devices.
 */
std::vector<std::size_t> getCachedBufferPoolCurrentSizeByDevices();

}  // namespace GPU
}  // namespace VideoStitch
