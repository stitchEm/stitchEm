// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/event.hpp"

#include <cuda_runtime.h>
#include <cuda/error.hpp>
#include <memory>

namespace VideoStitch {
namespace GPU {

class Event::DeviceEvent {
 public:
  explicit DeviceEvent(cudaEvent_t e) : event{new cudaEvent_t(e), &EventDeleter} {}

  static Event create(cudaEvent_t cle);

  static void EventDeleter(cudaEvent_t *cudaPtr) {
    CUDA_ERROR(cudaEventDestroy(*cudaPtr));
    delete cudaPtr;
  }

  friend class Event;

  std::shared_ptr<cudaEvent_t> event;
};

}  // namespace GPU
}  // namespace VideoStitch
