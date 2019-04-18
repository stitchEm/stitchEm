// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <libvideostitch/status.hpp>

#include <vector>

namespace VideoStitch {

enum {
  GPUHostAllocDefault = 0x00,      /**< Page-locked host memory */
  GPUHostAllocPinned = 0x01,       /**< Pinned host memory      */
  GPUHostAllocHostWriteOnly = 0x02 /**< Write-combined memory. WC memory is a good option for buffers
                                        that will be written by the CPU and read by the device
                                        via mapped pinned memory or host->device transfers.  */
};

namespace GPU {

// Backend specific pimpl class
class DeviceHostBuffer;

/** A chunk of memory on RAM (the host) allocated through the GPU API.
 *  Allocated page-locked by default to improve speed of memory copy operations
 *  between host and device.
 */
template <typename T>
class HostBuffer {
 public:
  // Default constructed, empty wrapper. Not a valid host buffer.
  HostBuffer() : pimpl(nullptr), hostData(nullptr), elements(0) {}

  // copy constructor & assignment operator
  HostBuffer(const HostBuffer&) = default;
  HostBuffer& operator=(const HostBuffer&) = default;

  // idem with move semantics
  HostBuffer(HostBuffer&& other)
      : pimpl(std::move(other.pimpl)), hostData(std::move(other.hostData)), elements(other.elements) {}
  HostBuffer& operator=(HostBuffer&& other) {
    pimpl = std::move(other.pimpl);
    hostData = std::move(other.hostData);
    elements = other.elements;
    return *this;
  }

  /**
   * Allocates memory on the host (RAM).
   * @param size number of elements of size T to allocate
   * @param name Name of the memory stats pool whose counter to increment.
   * @param flags Memory allocation flags. See CUDA, OpenCL documentations for performance considerations.
   * @return A valid GPU buffer on success, otherwise a failure Status.
   */
  static PotentialValue<HostBuffer<T>> allocate(size_t numElements, const char* name,
                                                unsigned int flags = GPUHostAllocDefault);

  /**
   * Frees the underlying host memory.
   * All wrapper objects referring to the host buffer become invalid.
   */
  Status release();

  size_t numElements() const { return elements; }

  size_t byteSize() const { return elements * sizeof(T); }

  T* hostPtr() const { return hostData; }

  operator T*() const { return hostPtr(); }

  /**
   * Conversion from read-write host buffer to read-only host buffer.
   * Only available on non-const host buffers.
   */
  HostBuffer<const T> as_const() const { return HostBuffer<const T>(pimpl, hostData, elements); }

  /**
   * Print name and size of currently allocated HostBuffers.
   */
  static void printPoolStats();

  /**
   * Total size in Bytes of currently allocated HostBuffers.
   */
  static std::size_t getPoolSize();

  /**
   * Total size in Bytes of currently allocated HostBuffers, by devices.
   */
  static std::vector<std::size_t> getPoolSizeByDevices();

  template <typename S>
  friend class HostBuffer;

  friend class DeviceHostBuffer;

 private:
  HostBuffer(DeviceHostBuffer* pimpl, T* d, size_t e) : pimpl(pimpl), hostData(d), elements(e) {}

  DeviceHostBuffer* pimpl;

  T* hostData;
  size_t elements;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  const DeviceHostBuffer& get();
};

}  // namespace GPU
}  // namespace VideoStitch
