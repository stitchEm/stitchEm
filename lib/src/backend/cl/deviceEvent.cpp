// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceEvent.hpp"

namespace VideoStitch {
namespace GPU {

Event::Event() : pimpl(nullptr) {}

Event::Event(DeviceEvent* pimpl) : pimpl(pimpl) {}

Event::Event(const Event& other) : pimpl(other.pimpl ? new DeviceEvent(*other.pimpl) : nullptr) {}

Event::~Event() {
  delete pimpl;
  pimpl = nullptr;
}

const Event::DeviceEvent& Event::get() const {
  assert(pimpl);
  return *pimpl;
}

Event Event::DeviceEvent::create(cl_event cle) {
  DeviceEvent* dve = new DeviceEvent(cle);
  return Event(dve);
}

Status Event::synchronize() {
  if (!pimpl) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Uninitialized GPU Event"};
  }
  return CL_ERROR(clWaitForEvents(1, &pimpl->event));
}

}  // namespace GPU
}  // namespace VideoStitch
