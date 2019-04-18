// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "buffer.hpp"
#include <cassert>

namespace VideoStitch {
namespace GPU {

/**
 * Base class for SharedPtrs
 */
template <typename T>
class SharedBuffer {
 public:
  /**
   * Creates a NULL pointer.
   */
  SharedBuffer()
      : buf(new Buffer<T>(), [](Buffer<T>* ptr) {
          if (ptr) {
            ptr->release();
            delete ptr;
          }
        }) {}

  SharedBuffer(const SharedBuffer& other) : buf(other.buf) {}

  SharedBuffer(SharedBuffer&& other) : buf(other.buf) {}

  /**
   * Check whether the buffer is valid
   */
  operator bool() const { return (buf.get() && (buf->byteSize() > 0)); }

  /**
   * Returns the const pointer but keeps ownership.
   */
  Buffer<const T> borrow_const() const { return *buf.get(); }

  /**
   * Returns the pointer but keeps ownership.
   */
  Buffer<T> borrow() const { return *buf.get(); }

  /**
   * Allocates a pointer.
   * @param size Size (in elements of type T).
   * @param name pool name.
   */
  Status alloc(size_t numElements, const char* name) {
    assert(!buf.get()->wasAllocated());
    // this seems like a dangerous mis-use of the API
    // enabling accidental reallocations
    // introduce a realloc() or something if this is really needed
    PotentialValue<Buffer<T>> potBuf = Buffer<T>::allocate(numElements, name);
    if (potBuf.ok()) {
      *(buf.get()) = potBuf.value();
    }
    return potBuf.status();
  }

 private:
  std::shared_ptr<Buffer<T>> buf;
};

}  // namespace GPU
}  // namespace VideoStitch
