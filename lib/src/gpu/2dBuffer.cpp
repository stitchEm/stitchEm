// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "2dBuffer.hpp"

namespace VideoStitch {
namespace GPU {

Buffer2D::Buffer2D() : pimpl(nullptr), width(0), height(0), pitch(0) {}

Buffer2D::Buffer2D(DeviceBuffer2D* pimpl, size_t width, size_t height, size_t pitch)
    : pimpl(pimpl), width(width), height(height), pitch(pitch) {}

const DeviceBuffer2D& Buffer2D::get() const {
  assert(pimpl);
  return *pimpl;
}
DeviceBuffer2D& Buffer2D::get() {
  assert(pimpl);
  return *pimpl;
}

}  // namespace GPU
}  // namespace VideoStitch
