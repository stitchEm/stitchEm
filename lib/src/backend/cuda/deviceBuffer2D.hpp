// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceBuffer.hpp"

namespace VideoStitch {
namespace GPU {

class DeviceBuffer2D : public DeviceBuffer<unsigned char> {
 public:
  explicit DeviceBuffer2D(unsigned char* ptr) : DeviceBuffer<unsigned char>(ptr) {}
};

}  // namespace GPU
}  // namespace VideoStitch
