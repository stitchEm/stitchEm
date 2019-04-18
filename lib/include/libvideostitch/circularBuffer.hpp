// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifndef CIRCULARBUFFER_HPP
#define CIRCULARBUFFER_HPP

#include "span.hpp"

#include <cassert>
#include <vector>
#include <algorithm>
#include <memory>

namespace VideoStitch {

/**
 * @brief A simple circular buffer class.
 *
 * Use this class to push and pop streams of data to a buffer.
 */
template <typename T>
class CircularBuffer {
 public:
  typedef std::pair<Span<T>, Span<T>> Slice;

  explicit CircularBuffer(size_t capacity = 0) : _begin(0), _capacity(capacity), _size(0), _storage(new T[capacity]) {}

  CircularBuffer(const CircularBuffer<T>& other)
      : _begin(other._begin), _capacity(other._capacity), _size(other._size), _storage(new T[_capacity]) {
    std::copy(other.storage_begin(), other.storage_end(), storage_begin());
  }

  bool empty() const { return _size == 0; }
  size_t size() const { return _size; }
  size_t capacity() const { return _capacity; }

  const T& operator[](size_t index) const { return *(storage_begin() + (_begin + index) % _capacity); }

  T& operator[](size_t index) { return const_cast<T&>((*const_cast<const CircularBuffer<T>*>(this))[index]); }

  /**
   * Pushes data to the end of the buffer, growing the storage if needed.
   * @param data Where to copy data from.
   * @param count Number of items to copy.
   */
  void push(const T* data, size_t size_to_copy) {
    if (size_to_copy > _capacity - _size) {
      // Grow storage if not enough space
      size_t growth = std::max(_capacity, size_to_copy);
      growth = std::max(growth, static_cast<std::size_t>(2));  // make sure it always grows at least by one element
      resize(growth + growth / 2);
    }

    size_t size_before_end = storage_end() - end();
    if (size_to_copy <= size_before_end) {
      // Write all at once when we do not pass the end
      // [b....         ]
      std::copy(data, data + size_to_copy, end());
      // [b....+++++    ]
    } else {
      // Write will pass the end, copy in two times
      // [    b.....    ]
      std::copy(data, data + size_before_end, end());
      data += size_before_end;
      // [    b....+++++]
      size_t size_left = size_to_copy - size_before_end;
      std::copy(data, data + size_left, storage_begin());
      // [++  b.........]
    }

    _size += size_to_copy;
  }

  /**
   * Pushes data to the end of the buffer, growing the storage if needed.
   * @param data Data to push.
   */
  void push(const T& data) { push(&data, 1); }

  /**
   * Pops data from the head of buffer.
   * @param data Where to copy data to.
   * @param count Number of items to copy.
   * @return Number of items effectively read.
   */
  size_t pop(T* data, size_t size_to_copy) {
    Slice s = slice(size_to_copy);
    std::copy(s.first.begin(), s.first.end(), data);
    if (!s.second.empty()) {
      data += s.first.size();
      std::copy(s.second.begin(), s.second.end(), data);
    }
    return erase(size_to_copy);
  }

  /**
   * Erases data from the head of the buffer.
   * @param count Number of items to erase.
   */
  size_t erase(size_t size_to_delete) {
    size_to_delete = std::min(size_to_delete, _size);
    _begin = (_begin + size_to_delete) % _capacity;
    _size -= size_to_delete;
    return size_to_delete;
  }

  /**
   * Clears the buffer.
   */
  void clear() {
    _begin = 0;
    _size = 0;
  }

  /**
   * Resizes and fills the buffer.
   * @param size New size of the buffer.
   * @param value Value to fill the buffer with.
   */
  void assign(size_t size, const T& value) {
    resize(size);
    std::fill(storage_begin(), storage_end(), value);
    _size = size;
  }

  /**
   * Returns a structure describing a slice of the buffer.
   * @param size Length of the slice.
   * @param offset Offset in the buffer. Defaults to 0.
   */
  Slice slice(size_t size_to_get, size_t offset = 0) {
    offset = std::min(offset, _size);
    size_to_get = std::min(size_to_get, _size - offset);
    T* begin = _storage.get() + (_begin + offset) % _capacity;
    size_t size_from_begin = storage_end() - begin;
    if (size_to_get <= size_from_begin) {
      // Read all at once when we're not reading past the end
      return std::make_pair(Span<T>(begin, begin + size_to_get), Span<T>());
    } else {
      // Read passes the end, copy in two times
      return std::make_pair(Span<T>(begin, begin + size_from_begin),
                            Span<T>(storage_begin(), storage_begin() + size_to_get - size_from_begin));
    }
  }

 private:
  void resize(size_t new_capacity) {
    std::unique_ptr<T> new_storage(new T[new_capacity]);
    size_t size = _size;
    if (_capacity > 0) {
      pop(new_storage.get(), size);
    }
    _storage.swap(new_storage);
    _begin = 0;
    _size = size;
    _capacity = new_capacity;
  }

  T* begin() { return _storage.get() + _begin; }
  const T* begin() const { return _storage.get() + _begin; }

  T* end() { return _storage.get() + (_begin + _size) % _capacity; }
  const T* end() const { return _storage.get() + (_begin + _size) % _capacity; }

  T* storage_begin() { return _storage.get(); }
  const T* storage_begin() const { return _storage.get(); }

  T* storage_end() { return _storage.get() + _capacity; }
  const T* storage_end() const { return _storage.get() + _capacity; }

  size_t _begin;
  size_t _capacity;
  size_t _size;
  std::unique_ptr<T> _storage;
};

}  // namespace VideoStitch

#endif  // CIRCULARBUFFER_HPP
