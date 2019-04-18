// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "buffer.hpp"
#include <cassert>

namespace VideoStitch {
namespace GPU {

/**
 * A UniqueBuffer is a smart wrapper for GPU Buffers that retains sole ownership
 * of GPU resources and frees the GPU memory when the UniqueBuffer goes out of scope.
 * No two UniqueBuffer instances can manage the same GPU memory.
 *
 * Reminder: the GPU Buffer is a dumb reference around GPU memory which requires manual memory
 * management (with alloc and release).
 *
 * The UniqueBuffer facilitates the memory management by automating the destruction,
 * similar to unique_ptr.
 */
template <typename T>
class UniqueBuffer {
 public:
  UniqueBuffer() : buf() {}

  /**
   * Takes ownership of the GPU Buffer, to release its memory when UniqueBuffer is destroyed
   */
  explicit UniqueBuffer(Buffer<T> buf) : buf(buf) {}

  UniqueBuffer(const UniqueBuffer& other) = delete;

  UniqueBuffer(UniqueBuffer&& other) {
    if (hasValidBuffer) {
      buf.release();
    }

    buf = other.buf;
    hasValidBuffer = other.hasValidBuffer;

    other.buf = Buffer<T>();
    other.hasValidBuffer = false;
  }

  /** Move assignment operator */
  UniqueBuffer& operator=(UniqueBuffer&& other) {
    if (hasValidBuffer) {
      buf.release();
    }

    this->buf = other.buf;
    this->hasValidBuffer = other.hasValidBuffer;

    other.buf = Buffer<T>();
    other.hasValidBuffer = false;

    return *this;
  }

  operator bool() const { return buf.byteSize() > 0; }

  /**
   * Returns the GPU Buffer but keeps ownership
   */
  Buffer<T> borrow() const { return buf; }

  /**
   * Returns the const GPU Buffer but keeps ownership
   */
  Buffer<const T> borrow_const() const { return buf; }

  /**
   * Releases ownership
   */
  Buffer<T> releaseOwnership() {
    Buffer<T> res = buf;
    buf = Buffer<T>();
    hasValidBuffer = false;
    return res;
  }

  /**
   * Allocates a GPU Buffer for the first time after creating the object.
   * Allocating a UniqueBuffer that is already allocated is an error, use .recreate()
   * @param size Size (in elements of type T).
   * @param name pool name.
   */
  Status alloc(size_t numElements, const char* name) {
    if (hasValidBuffer) {
      return {Origin::GPU, ErrType::ImplementationError,
              "Attempting to allocate a UniqueBuffer that is already allocated"};
    }

    PotentialValue<Buffer<T>> potBuf = Buffer<T>::allocate(numElements, name);
    if (potBuf.ok()) {
      buf = potBuf.value();
      hasValidBuffer = true;
    }
    return potBuf.status();
  }

  /**
   * Replaces current buffer with a new GPU Buffer of different size
   * Calling recreate on an uninitialized unique buffer is a valid operation.
   *
   * Note: the content of the GPU memory is not preserved, a new buffer is created
   *
   * @param size Size (in elements of type T).
   * @param name pool name.
   */
  Status recreate(size_t numElements, const char* name) {
    if (hasValidBuffer) {
      buf.release();
      hasValidBuffer = false;
    }

    return this->alloc(numElements, name);
  }

  ~UniqueBuffer() {
    if (hasValidBuffer) {
      buf.release();
    }
  }

 private:
  Buffer<T> buf;
  bool hasValidBuffer = false;
};

template <typename T>
class PotentialUniqueBuffer : public PotentialValue<UniqueBuffer<T>> {
 public:
  PotentialUniqueBuffer(const Status& status) : PotentialValue<UniqueBuffer<T>>(status) {}
  PotentialUniqueBuffer(UniqueBuffer<T>&& value) : PotentialValue<UniqueBuffer<T>>(std::move(value)) {}
  PotentialUniqueBuffer(const PotentialUniqueBuffer&& other) = delete;
  PotentialUniqueBuffer(PotentialUniqueBuffer&& other) : PotentialValue<UniqueBuffer<T>>(std::move(other)) {}

  Buffer<T> borrow() const {
    const UniqueBuffer<T>& ub = this->value_;
    return ub.borrow();
  }

  Buffer<const T> borrow_const() const {
    const UniqueBuffer<T>& ub = this->value_;
    return ub.borrow_const();
  }

  Buffer<T> releaseOwnership() {
    UniqueBuffer<T>& ub = this->value_;
    return ub.releaseOwnership();
  }
};

/**
 * Helper function to create UniqueBuffers with a single call
 * Name doesn't start with 'allocate' to make it stand out in the source code
 * as it has a different lifetime
 */
template <typename T>
PotentialUniqueBuffer<T> uniqueBuffer(size_t numElements, const char* name) {
  UniqueBuffer<T> buf;
  const Status allocationStatus = buf.alloc(numElements, name);
  if (!allocationStatus.ok()) {
    return PotentialUniqueBuffer<T>(allocationStatus);
  }
  return PotentialUniqueBuffer<T>(std::move(buf));
}

}  // namespace GPU
}  // namespace VideoStitch
