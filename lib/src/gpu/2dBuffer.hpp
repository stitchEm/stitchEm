// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace GPU {

class DeviceBuffer2D;

/** Wraps a backend specific pointer to a chunk of 2D memory on the GPU.
 *  The purpose of 2D memory is to enable coalesced reads and writes.
 *
 *  The lifetime of the GPU memory is independent of the wrapper objects.
 *  It is managed manually with allocate() and release().
 */
class Buffer2D {
 public:
  /**
   * Allocates memory on the GPU.
   * @param size number of bytes to allocate
   * @param name Name of the memory stats pool whose counter to increment.
   * @return A valid GPU buffer on success, otherwise a failure Status.
   */
  static PotentialValue<Buffer2D> allocate(size_t width, size_t height, const char* name);

  /**
   * Frees the underlying GPU memory.
   * All wrapper objects referring to the buffer become invalid.
   */
  Status release() const;

  void* devicePtr() const;

  // Default constructed, empty Buffer wrapper. Not a valid GPU buffer.
  Buffer2D();
  ~Buffer2D();

  Buffer2D(const Buffer2D& other);
  void swap(Buffer2D& other) {
    std::swap(pimpl, other.pimpl);
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(pitch, other.pitch);
  }

  Buffer2D operator=(Buffer2D other) {
    swap(other);
    return *this;
  }

  bool operator==(const Buffer2D& other) const;

  bool operator!=(const Buffer2D& other) const { return !(*this == other); }

  friend class DeviceBuffer2D;

  size_t getWidth() const { return width; }
  size_t getHeight() const { return height; }
  size_t getPitch() const { return pitch; }

 private:
  Buffer2D(DeviceBuffer2D* pimpl, size_t width, size_t height, size_t pitch);

  DeviceBuffer2D* pimpl;

  size_t width;   // in bytes
  size_t height;  // in bytes
  size_t pitch;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  const DeviceBuffer2D& get() const;
  DeviceBuffer2D& get();
};

}  // namespace GPU
}  // namespace VideoStitch
