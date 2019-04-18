// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/event.hpp"
#include "cl_error.hpp"
#include "opencl.h"

namespace VideoStitch {
namespace GPU {

class Event::DeviceEvent {
 public:
  explicit DeviceEvent(cl_event e) : event(e) {}

  DeviceEvent(const DeviceEvent& other) : event(other.event) {
    if (event) {
      clRetainEvent(event);
    }
  }

  ~DeviceEvent() {
    if (event) {
      clReleaseEvent(event);
    }
  }

  static Event create(cl_event cle);

  cl_event event = nullptr;
};

}  // namespace GPU
}  // namespace VideoStitch
