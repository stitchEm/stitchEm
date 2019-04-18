// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/gpu_device.hpp"

#include "gpu/buffer.hpp"

#include "context.hpp"

#include <libgpudiscovery/genericDeviceInfo.hpp>

#include <algorithm>
#include <functional>

namespace VideoStitch {
namespace GPU {

std::atomic<int> defaultDevice(-1);

Status checkDefaultBackendDeviceInitialization() {
  const auto& contextStatus = getContext();
  return contextStatus.status();
}

Status setDefaultBackendDeviceVS(int vsDevice) {
  int device;
  if (Discovery::getBackendDeviceIndex(vsDevice, device)) {
    defaultDevice = device;
    return Status::OK();
  }
  return {Origin::GPU, ErrType::ImplementationError, "[OpenCL] trying to set a device that does not exist"};
}

Status getDefaultBackendDeviceContext(void*) {
  assert(0);
  return {Origin::GPU, ErrType::UnsupportedAction, "[OpenCL] getDefaultDeviceContext not implemented"};
}

Status useDefaultBackendDevice() {
  if (defaultDevice == -1) {
    return {Origin::GPU, ErrType::ImplementationError, "[OpenCL] default device is not set"};
  }
  return Status::OK();
}

Status setDefaultBackendDevice(int device) {
  if (device >= Discovery::getNumberOfOpenCLDevices() || device < 0) {
    return {Origin::GPU, ErrType::ImplementationError, "[OpenCL] trying to set a non-existent device"};
  }
  if (defaultDevice != -1 && device != defaultDevice) {
    return {Origin::GPU, ErrType::ImplementationError, "[OpenCL] changing the default device is not permitted"};
  }
  defaultDevice = device;
  return Status::OK();
}

Status getDefaultBackendDevice(int* device) {
  if (defaultDevice == -1) {
    return {Origin::GPU, ErrType::ImplementationError, "[OpenCL] default device is not set"};
  }
  *device = defaultDevice;
  return Status::OK();
}

Discovery::Framework getFramework() { return Discovery::Framework::OpenCL; }
PotentialValue<size_t> getMemoryUsage() {
  size_t used_memory = 0;
  used_memory += getBufferPoolCurrentSize();
  // used_memory += getCachedBufferPoolCurrentSize();
  return PotentialValue<size_t>(used_memory);
}

PotentialValue<std::vector<size_t> > getMemoryUsageByDevices() {
  // add buffers, cached buffers and allocated size on device
  // initialize with first buffers
  std::vector<size_t> used_memory = getBufferPoolCurrentSizeByDevices();
  // accumulate other buffers
  // std::transform(used_memory.begin(), used_memory.end(), getCachedBufferPoolCurrentSizeByDevices().begin(),
  // used_memory.begin(), std::plus<size_t>());
  return PotentialValue<std::vector<size_t> >(used_memory);
}

}  // namespace GPU
}  // namespace VideoStitch
