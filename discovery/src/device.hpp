// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libgpudiscovery/cudaDeviceInfo.hpp"

#ifdef OPENCL_FOUND
#include "libgpudiscovery/openCLDeviceInfo.hpp"
#endif

#include "libgpudiscovery/genericDeviceInfo.hpp"

namespace VideoStitch {
namespace Discovery {

class DevicesInfo {
 public:
  int getNumberOfDevices() const;
  int getNumberOfOpenCLDevices() const;
  int getNumberOfCudaDevices() const;
  bool getDeviceProperties(unsigned vsDeviceIndex, DeviceProperties&) const;
  bool getCudaDeviceProperties(unsigned vsDeviceIndex, DeviceProperties&) const;
#ifdef OPENCL_FOUND
  bool getOpenCLDeviceProperties(unsigned vsDeviceIndex, OpenCLDevice&) const;
#endif
  bool getBackendDeviceIndex(int vsDeviceIndex, int& backendDeviceIndex) const;
  bool getVSDeviceIndex(int backendDeviceIndex, int& vsDeviceIndex, Framework framework) const;
  FrameworkStatus getFrameworkStatus(Framework framework) const;

  static const DevicesInfo& getInstance();

 private:
  DevicesInfo();
  void collectOpenCLDeviceInfo();
  void collectCudaDeviceInfo();
  void collectGenericDeviceInfo();
  std::vector<DeviceProperties> cudaDevices;
#ifdef OPENCL_FOUND
  std::vector<OpenCLDevice> openCLDevices;
#endif
  std::vector<DeviceProperties> genericDevices;
  FrameworkStatus cudaStatus;
  FrameworkStatus openCLStatus;
};

}  // namespace Discovery
}  // namespace VideoStitch
