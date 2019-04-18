// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// NTV2 framework includes ksmedia.hpp, which defines the speaker map with the same names...

#include "libvideostitch/audio.hpp"

#include "ntv2plugin.hpp"

#include "ntv2devicescanner.h"
#include "ajastuff/system/process.h"

namespace VideoStitch {

NTV2Device::NTV2Device() : deviceID(DEVICE_ID_NOTFOUND), initialized(false) {}

NTV2Device::~NTV2Device() {
  if (initialized) {
    device.ReleaseStreamForApplication(appSignature, static_cast<uint32_t>(AJAProcess::GetPid()));
    device.SetEveryFrameServices(savedTaskMode);
  }
}

NTV2Device* NTV2Device::getDevice(uint32_t deviceIndex) {
  std::unique_lock<std::mutex> lk(registryMutex);
  if (registry.find(deviceIndex) == registry.end()) {
    NTV2Device* device = new NTV2Device();
    if (AJA_FAILURE(device->init(deviceIndex))) {
      delete device;
      return nullptr;
    }
    registry[deviceIndex] = device;
  }
  return registry[deviceIndex];
}

AJAStatus NTV2Device::init(uint32_t deviceIndex) {
  // Open the device
  if (!CNTV2DeviceScanner::GetDeviceAtIndex(deviceIndex, device)) {
    return AJA_STATUS_OPEN;
  }

  if (!device.AcquireStreamForApplication(appSignature, static_cast<uint32_t>(AJAProcess::GetPid()))) {
    return AJA_STATUS_BUSY;  //  Another app is using the device
  }

  device.GetEveryFrameServices(&savedTaskMode);
  device.SetEveryFrameServices(NTV2_OEM_TASKS);
  deviceID = device.GetDeviceID();
  return AJA_STATUS_SUCCESS;
}

}  // namespace VideoStitch
