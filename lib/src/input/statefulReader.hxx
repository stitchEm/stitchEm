// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "statefulReader.hpp"

#include "libvideostitch/gpu_device.hpp"

#include <cassert>

namespace VideoStitch {
namespace Input {

template <typename StateT>
StatefulReader<StateT>::StatefulReader(int64_t width, int64_t height,
                                       int64_t frameDataSize,
                                       PixelFormat format, AddressSpace addressSpace, FrameRate frameRate,
                                       int firstFrame, int lastFrame, bool isProcedural,
                                       const unsigned char* maskHostBuffer, int flags)
  : Reader(-1), // never called : https://isocpp.org/wiki/faq/multiple-inheritance#virtual-inheritance-ctors
    VideoReader(width, height, frameDataSize, format, addressSpace, frameRate, firstFrame, lastFrame,
                isProcedural, maskHostBuffer, flags) {
}

template <typename StateT>
StatefulReader<StateT>::~StatefulReader() {
  assert(perDeviceData.empty());  // Did override perThreadCleanup and forgot to base-call ?
}

template <typename StateT>
const StateT* StatefulReader<StateT>::getCurrentDeviceData() const {
  std::unique_lock<std::mutex> lock(mutex);
  int dev = getCurrentDevice();
  typename std::map<int, StateT>::const_iterator it = perDeviceData.find(dev);
  if (it == perDeviceData.end()) {
    return NULL;
  }
  return &(it->second);
}

template <typename StateT>
Status StatefulReader<StateT>::setCurrentDeviceData(const StateT& state) {
  int device = getCurrentDevice();
  if (device < 0) {
    return { Origin::GPU, ErrType::SetupFailure, "Invalid GPU device number encountered: " + std::to_string(device) };
  }
  std::unique_lock<std::mutex> lock(mutex);
  perDeviceData[device] = state;
  return Status::OK();
}

template <typename StateT>
void StatefulReader<StateT>::perThreadCleanup() {
  perDeviceData.erase(getCurrentDevice());
}

template <typename StateT>
int StatefulReader<StateT>::getCurrentDevice() const {
  int device = -1;
  GPU::getDefaultBackendDevice(&device);
  return device;
}

}
}

