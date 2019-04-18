// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "typeTraits.hpp"
#include "vectorTypes.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/logging.hpp"

#include <sstream>
#include <vector>

namespace VideoStitch {
namespace GPU {

template <typename T>
class DeviceBuffer;

/** Wraps a backend specific pointer to a chunk of memory on the GPU.
 *  The lifetime of the GPU memory is independent of the wrapper objects.
 *  It is managed manually with allocate() and release().
 *
 *  Currently supported underlying data types:
 *  char, unsigned char, uchar3
 *  unsigned short
 *  int, unsigned int (uint32_t)
 *  float, float2, float4
 *
 *  More types can be added, but need to be explicitly instantiated in the backend.
 */
template <typename T>
class Buffer {
 public:
  /**
   * Allocates memory on the GPU.
   * @param size number of elements of size T to allocate
   * @param name Name of the memory stats pool whose counter to increment.
   * @return A valid GPU buffer on success, otherwise a failure Status.
   */
  static PotentialValue<Buffer> allocate(size_t numElements, const char* name);

  /**
   * Frees the underlying GPU memory.
   * All wrapper objects referring to the buffer become invalid.
   */
  Status release() const;

  // TODO should create PotentialBuffer
  // TODO deprecate. either use separate buffers or do pointer arithmetic in kernel
  Buffer createSubBuffer(size_t elementOffset);

  size_t byteSize() const { return sizeof(T) * elements; }

  size_t numElements() const { return elements; }

  /** Was allocate ever called on this wrapper?
   *  Note: no warranty given that it is still alive.
   *  It could have been destroyed/released in the meantime.
   */
  bool wasAllocated() const;

  template <typename S>
  Buffer<S> as() const {
    if (elements * sizeof(T) % sizeof(S) != 0) {
      std::stringstream s;
      s << "Typecasting buffer with " << elements << " elements of " << sizeof(T) << " bytes per element to "
        << sizeof(S) << " bytes per element." << std::endl;
      Logger::get(Logger::Warning) << s.str();
    }
    size_t elem_as = elements * sizeof(T) / sizeof(S);
    return Buffer<S>(pimpl, elem_as);
  }

  /**
   * Conversion from read-write Buffer to read-only Buffer.
   * Only available on non-const Buffers.
   */
  CLASS_S_EQ_T_ENABLE_IF_S_NON_CONST
  Buffer<const S> as_const() const { return Buffer<const S>(pimpl, elements); }

  /**
   * Provide automatic conversion into read-only buffer, like a raw pointer type would.
   * This is usually needed when passing Buffer<T> into functions parametrized with Buffer<const T>.
   */
  CLASS_S_EQ_T_ENABLE_IF_S_NON_CONST
  operator Buffer<const S>() const { return Buffer<const S>(pimpl, elements); }

  // Default constructed, empty Buffer wrapper. Not a valid GPU buffer.
  Buffer();
  ~Buffer();

  Buffer(const Buffer& other);
  void swap(Buffer& other) {
    std::swap(pimpl, other.pimpl);
    std::swap(isSubBuffer, other.isSubBuffer);
    std::swap(elements, other.elements);
  }

  Buffer operator=(Buffer other) {
    swap(other);
    return *this;
  }

  T* devicePtr() const;
  static Buffer wrap(T* devicePtr, size_t num);

  bool operator==(const Buffer& other) const;

  bool operator!=(const Buffer& other) const { return !(*this == other); }

  template <typename S>
  friend class Buffer;

  friend class DeviceBuffer<T>;

 private:
  Buffer(DeviceBuffer<T>* pimpl, size_t num, bool isSubBuffer);
  Buffer(DeviceBuffer<T>* pimpl, size_t num);

  template <typename S>
  Buffer(DeviceBuffer<S>* pimpl, size_t num);

  DeviceBuffer<T>* pimpl;

  size_t elements;
  bool isSubBuffer = false;

 public:
  /** Provide the GPU backend implementation with simple access to the underlying data structure. */
  const DeviceBuffer<T>& get() const;
  DeviceBuffer<T>& get();
};

}  // namespace GPU

/**
 * Print name and size of currently allocated Buffers.
 */
void printBufferPoolStats();

/**
 * Total size in Bytes of currently allocated Buffers.
 */
std::size_t getBufferPoolCurrentSize();

/**
 * Total size in Bytes of currently allocated Buffers, by devices.
 */
std::vector<std::size_t> getBufferPoolCurrentSizeByDevices();

}  // namespace VideoStitch
