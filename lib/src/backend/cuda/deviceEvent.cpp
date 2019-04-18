// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceEvent.hpp"

#include "deviceStream.hpp"

#include "cuda/error.hpp"

namespace VideoStitch {
namespace GPU {

Event::Event() : pimpl(nullptr) {}

Event::Event(DeviceEvent* pimpl) : pimpl(pimpl) {}

Event::Event(const Event& other) : pimpl(other.pimpl ? new DeviceEvent(*other.pimpl) : nullptr) {}

Event::~Event() {
  delete pimpl;
  pimpl = nullptr;
}

Status Event::synchronize() { return CUDA_ERROR(cudaEventSynchronize(*(pimpl->event))); }

Event Event::DeviceEvent::create(cudaEvent_t cudaEvent) {
  DeviceEvent* dve = new DeviceEvent(cudaEvent);
  return Event(dve);
}

const Event::DeviceEvent& Event::get() const {
  assert(pimpl);
  return *pimpl;
}

}  // namespace GPU
}  // namespace VideoStitch
