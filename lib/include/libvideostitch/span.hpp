// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#ifndef SPAN_HPP
#define SPAN_HPP

namespace VideoStitch {

/**
 * @brief A class to describe a span of an array
 */
template <typename T>
class Span {
  T* _begin;
  T* _end;

 public:
  Span(T* b = nullptr, T* e = nullptr) : _begin(b), _end(e) {}
  Span(T* b, size_t size) : _begin(b), _end(b + size) {}

  bool empty() const { return _begin == _end; }

  size_t size() const { return _end - _begin; }

  T* begin() { return _begin; }

  T* data() { return _begin; }

  T* end() { return _end; }

  const T* begin() const { return _begin; }

  const T* data() const { return _begin; }

  const T* end() const { return _end; }

  const T& operator[](size_t pos) const { return _begin[pos]; }

  T& operator[](size_t pos) { return _begin[pos]; }
};

}  // namespace VideoStitch

#endif  // SPAN_HPP
