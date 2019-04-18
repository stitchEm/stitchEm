// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceStream.hpp"

#include "deviceEvent.hpp"

#include "gpu/stream.hpp"

#include "context.hpp"

// if this is enabled, there will be a single command queue
// #define DEBUG_SYNC_GPU_SINGLE_STREAM

namespace VideoStitch {
namespace GPU {

Stream::Stream() : pimpl(nullptr) {}

Status Stream::destroy() {
  if (!pimpl) {
    return {Origin::GPU, ErrType::ImplementationError, "Attempting to destroy uninitialized GPU Stream"};
  }
  cl_command_queue queue = *pimpl;
  if (!queue) {
    return {Origin::GPU, ErrType::ImplementationError, "Attempting to release uninitialized command queue"};
  }

  Status status = CL_ERROR(clReleaseCommandQueue(*pimpl));

#ifndef DEBUG_SYNC_GPU_SINGLE_STREAM
  delete pimpl;
  pimpl = nullptr;
#endif
  return status;
}

Status Stream::flush() const { return CL_ERROR(clFlush(*pimpl)); }

Status Stream::synchronize() const { return CL_ERROR(clFinish(*pimpl)); }

PotentialValue<Event> Stream::recordEvent() const {
  cl_event cle;
  PROPAGATE_CL_ERR(clEnqueueMarkerWithWaitList(get(), 0, nullptr, &cle));
  return Event::DeviceEvent::create(cle);
}

#if (_MSC_VER && _MSC_VER < 1900)
// C++11 magic statics support from Visual Studio 2015
static std::mutex defaultStreamInitMutex;
#endif

Stream Stream::getDefault() {
#if (_MSC_VER && _MSC_VER < 1900)
  std::lock_guard<std::mutex> initLock(defaultStreamInitMutex);
#endif
  static Stream stream = []() {
    auto potStream = DeviceStream::createPotentialStream();
    assert(potStream.ok());
    return potStream.value();
  }();
  return stream;
}

const Stream::DeviceStream& Stream::get() const {
  assert(pimpl);
  return *pimpl;
}

PotentialValue<Stream> Stream::DeviceStream::createPotentialStream() {
  const auto& potContext = getContext();
  FAIL_RETURN(potContext.status());
  int err;
  cl_command_queue commands = clCreateCommandQueue(potContext.value(), potContext.value().deviceID(), 0, &err);
  Status status = CL_ERROR(err);
  if (status.ok()) {
    auto stream = Stream();
    delete stream.pimpl;
    stream.pimpl = new DeviceStream(commands);
    return stream;
  }
  return status;
}

Status Stream::waitOnEvent(Event event) const {
  return CL_ERROR(clEnqueueBarrierWithWaitList(get(), 1, &event.get().event, nullptr));
}

PotentialValue<Stream> Stream::create() {
#ifdef DEBUG_SYNC_GPU_SINGLE_STREAM
  clRetainCommandQueue(GPU::Stream::getDefault().get());
  return PotentialValue<Stream>(GPU::Stream::getDefault());
#else
  return DeviceStream::createPotentialStream();
#endif
}

bool Stream::operator==(const Stream& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

}  // namespace GPU
}  // namespace VideoStitch
