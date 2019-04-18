// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/stream.hpp"

#include "deviceStream.hpp"
#include "deviceEvent.hpp"

#include "cuda/error.hpp"

#if (_MSC_VER && _MSC_VER < 1900)
#include <mutex>
#endif

#include <cuda_runtime.h>

namespace VideoStitch {
namespace GPU {

Stream::Stream() : pimpl(nullptr) {}

Stream::Stream(cudaStream_t cudaStream) : Stream() {
  delete pimpl;
  pimpl = new DeviceStream(cudaStream);
}

void Stream::destroyDeprecatedCUDAWrapper() {
  delete pimpl;
  pimpl = nullptr;
}

#if (_MSC_VER && _MSC_VER < 1900)
// C++11 magic statics supported in Visual Studio 2015 and later
static std::mutex defaultStreamInitMutex;
#endif

Stream Stream::getDefault() {
#if (_MSC_VER && _MSC_VER < 1900)
  std::lock_guard<std::mutex> initLock(defaultStreamInitMutex);
#endif
  // CUDA default Stream is NULL, keep it that way for compatibility
  // might in the future just create any stream and merge implementation
  // with OpenCL
  cudaStream_t s = NULL;
  static Stream defaultStream = Stream(s);
  return defaultStream;
}

PotentialValue<Stream> Stream::create() {
  cudaStream_t cuStream;
  auto status = CUDA_ERROR(cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking));
  if (status.ok()) {
    auto stream = Stream();
    delete stream.pimpl;
    stream.pimpl = new DeviceStream(cuStream);
    return PotentialValue<Stream>(stream);
  }
  return PotentialValue<Stream>(status);
}

Status Stream::destroy() {
  assert(*this != getDefault());
  if (!pimpl) {
    return {Origin::GPU, ErrType::ImplementationError, "Trying to destroy an uninitialized GPU Stream"};
  }
  auto status = CUDA_ERROR(cudaStreamDestroy(*pimpl));
  delete pimpl;
  pimpl = nullptr;
  return status;
}

PotentialValue<Event> Stream::recordEvent() const {
  cudaEvent_t event;
  FAIL_RETURN(CUDA_ERROR(cudaEventCreate(&event, cudaEventBlockingSync | cudaEventDisableTiming)));
  FAIL_RETURN(CUDA_ERROR(cudaEventRecord(event, get())));
  return Event::DeviceEvent::create(event);
}

Status Stream::synchronize() const {
  // Source tab on Vahana, on the stitching box (at least on the surface prototype, which is much better at GPU
  // occupancy), was yielding a 200% CPU usage, keeping 2 cores busy for nothing. The software was busy-waiting for the
  // GPU to complete its work, burning a lot of CPU in the process. Forcing waiting on event instead of the whole stream
  // seems like a hack, but the CPU usage dropped to 15%. Events have a fine granularity with regard to interaction with
  // the OS scheduler. Created with the BlockingSync flag, they will wait instead of spinning. This theoretically
  // increases the latency (think the order of a context-switch duration), but the trade-off is the CPU is completely
  // free to do something else in the meantime.
  cudaEvent_t event;
  FAIL_RETURN(CUDA_ERROR(cudaEventCreate(&event, cudaEventBlockingSync | cudaEventDisableTiming)));
  FAIL_RETURN(CUDA_ERROR(cudaEventRecord(event, get())));
  FAIL_RETURN(CUDA_ERROR(cudaEventSynchronize(event)));
  return CUDA_ERROR(cudaEventDestroy(event));
}

Status Stream::flush() const { return Status::OK(); }

Status Stream::waitOnEvent(Event event) const {
  return CUDA_ERROR(cudaStreamWaitEvent(get(), *(event.get().event), 0));
}

const Stream::DeviceStream& Stream::get() const {
  // Every Stream needs to be initialized properly
  assert(pimpl);
  return *pimpl;
}

bool Stream::operator==(const Stream& other) const {
  if (pimpl && other.pimpl) {
    return *pimpl == *other.pimpl;
  }
  return !pimpl && !other.pimpl;
}

}  // namespace GPU
}  // namespace VideoStitch
