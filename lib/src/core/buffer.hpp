// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"

#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"

namespace VideoStitch {
namespace Core {

class Buffer {
 public:
  explicit Buffer(GPU::HostBuffer<unsigned char> host) : host(host), device(), address(Host), initialized(true) {}
  explicit Buffer(GPU::Buffer<unsigned char> device) : host(), device(device), address(Device), initialized(true) {}
  Buffer() : host(), device(), address(), initialized(false) {}

  bool operator==(const Buffer& o) const {
    if (initialized != o.initialized) {
      return false;
    }

    if (!initialized) {
      return true;
    }

    if (address != o.address) {
      return false;
    }

    switch (address) {
      case Host:
        return host == o.host;
      case Device:
        return device == o.device;
    }

    assert(false);
    return false;
  }

  AddressSpace addressSpace() const { return address; }

  GPU::HostBuffer<unsigned char> hostBuffer() const {
    assert(address == Host);
    return host;
  }

  GPU::Buffer<unsigned char> deviceBuffer() const {
    assert(address == Device);
    return device;
  }

  void release() {
    if (!initialized) {
      return;
    }
    switch (address) {
      case Host:
        host.release();
        break;
      case Device:
        device.release();
        break;
    }
  }

  unsigned char* rawPtr() const {
    if (!initialized) {
      return nullptr;
    }

    switch (address) {
      case Host:
        return hostBuffer().hostPtr();
      case Device:
        return (unsigned char*)deviceBuffer().devicePtr();
    }

    assert(false);
    return nullptr;
  }

 private:
  GPU::HostBuffer<unsigned char> host;
  GPU::Buffer<unsigned char> device;
  AddressSpace address;
  bool initialized;
};

}  // namespace Core
}  // namespace VideoStitch
