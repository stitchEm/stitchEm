// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "opencl.h"

namespace VideoStitch {
namespace GPU {

class DeviceBuffer2D {
 public:
  explicit DeviceBuffer2D(cl_mem im) : image(im) {}

  cl_mem& raw() { return image; }

  operator cl_mem() const { return image; }

  operator cl_mem&() { return image; }

  bool operator==(const DeviceBuffer2D& other) const { return image == other.image; }

 private:
  cl_mem image;

  friend class Buffer2D;
};

}  // namespace GPU
}  // namespace VideoStitch
