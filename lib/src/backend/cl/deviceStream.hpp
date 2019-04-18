// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/stream.hpp>
#include "opencl.h"

namespace VideoStitch {
namespace GPU {

class Stream::DeviceStream {
 public:
  DeviceStream() : commandQueue(nullptr) {}

  explicit DeviceStream(cl_command_queue cq) : commandQueue(cq) {}

  static PotentialValue<Stream> createPotentialStream();

  operator cl_command_queue() const { return commandQueue; }

  bool operator==(const DeviceStream& other) const { return commandQueue == other.commandQueue; }

  bool operator!=(const DeviceStream& other) const { return !(*this == other); }

 private:
  cl_command_queue commandQueue;
};

}  // namespace GPU
}  // namespace VideoStitch
