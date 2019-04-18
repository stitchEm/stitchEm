// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "rtmpEnums.hpp"
#include "libvideostitch/config.hpp"

#include <mutex>
#include <vector>
#include <stdint.h>
#include <memory>
#include <cstring>

namespace VideoStitch {
namespace IO {
struct DataPacket {
  typedef std::shared_ptr<unsigned char> Storage;

  DataPacket(size_t size = 0)
      : _storage(new unsigned char[size], std::default_delete<unsigned char[]>()), _size(size) {}

  DataPacket(const unsigned char* data, size_t size)
      : _storage(new unsigned char[size], std::default_delete<unsigned char[]>()), _size(size) {
    copy(data, size);
  }

  DataPacket(Storage storage, size_t size) : _storage(storage), _size(size) {}

  DataPacket(const std::vector<unsigned char>& other)
      : _storage(new unsigned char[other.size()], std::default_delete<unsigned char[]>()), _size(other.size()) {
    copy(other.data(), other.size());
  }

  size_t size() const { return _size; }

  unsigned char operator[](size_t pos) const { return _storage.get()[pos]; }

  unsigned char& operator[](size_t pos) { return _storage.get()[pos]; }

  unsigned char* data() const { return _storage.get(); }

  unsigned char* begin() { return _storage.get(); }

  unsigned char* end() { return _storage.get() + _size; }

  const unsigned char* begin() const { return _storage.get(); }

  const unsigned char* end() const { return _storage.get() + _size; }

  Storage storage() { return _storage; }

  static Storage make_storage(size_t size) {
    return Storage(new unsigned char[size], std::default_delete<unsigned char[]>());
  }

  mtime_t timestamp;  ///> in milliseconds
  PacketType type;

 private:
  void copy(const unsigned char* data, size_t size) { memcpy(_storage.get(), data, size); }

  Storage _storage;
  size_t _size;
};

struct ColorDescription {
  int fullRange;
  int primaries;
  int transfer;
  int matrix;
};

extern std::mutex rtmpInitMutex;
}  // namespace IO
}  // namespace VideoStitch
