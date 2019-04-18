// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "config.hpp"
#include <vector>
#include <iostream>
#include <type_traits>

namespace VideoStitch {
namespace Discovery {

enum class Framework { CUDA, OpenCL, Unknown };

enum class FrameworkStatus : int {
  Ok = 0x00,
  OutdatedDriver = 0x01,
  MissingDriver = 0x02,
  NoCompatibleDevice = 0x03,
  GenericError = 0xFF
};

enum class DeviceType { GPU, CPU, other };

enum class Vendor { AMD, NVIDIA, INTEL, UNKNOWN };

struct DeviceProperties {
  /** Name */
  char name[256];

  /** Driver Version */
  char driverVersion[256];

  /** 32 or 64 bit pointers */
  unsigned addressBits;

  /** Size in bytes */
  size_t globalMemSize;

  /** Device type */
  DeviceType type;

  /** Vendor name */
  Vendor vendor;

  /** Can be used to stitch videos? */
  bool compatible;

  Framework supportedFramework;
};

inline std::ostream& operator<<(std::ostream& os, const Discovery::DeviceProperties& prop) {
  os << "GPU device: " << prop.name << " (" << prop.addressBits << " Bit ";

  switch (prop.type) {
    case DeviceType::GPU:
      os << "GPU";
      break;
    case DeviceType::CPU:
      os << "CPU";
      break;
    case DeviceType::other:
      os << "other";
      break;
  }

  os << "), global memory size: " << prop.globalMemSize / 1024 / 1024 << " MB";
  return os;
}

VS_DISCOVERY_EXPORT int getNumberOfDevices();
VS_DISCOVERY_EXPORT int getNumberOfOpenCLDevices();
VS_DISCOVERY_EXPORT int getNumberOfCudaDevices();

VS_DISCOVERY_EXPORT FrameworkStatus getFrameworkStatus(Framework framework);

// returns true and fills the 2nd argument if a device corresponds to the vsDeviceIndex
// Otherwise, it returns false and doesn't fill the 2nd argument
VS_DISCOVERY_EXPORT bool getDeviceProperties(unsigned vsDeviceIndex, struct DeviceProperties& prop);

// returns true and fills the 2nd argument with the corresponding backend index
// if a device corresponds to the vsDeviceIndex.
// Otherwise, it returns false and doesn't fill the 2nd argument
VS_DISCOVERY_EXPORT bool getBackendDeviceIndex(int vsDeviceIndex, int& backendDeviceIndex);

// returns true and fills the 2nd argument with the corresponding videostitch index
// if a device corresponds to the backendIndex of the given famework.
// Otherwise, it returns false and doesn't fill the 2nd argument
VS_DISCOVERY_EXPORT bool getVSDeviceIndex(int backendDeviceIndex, int& vsDeviceIndex, Framework framework);

VS_DISCOVERY_EXPORT std::string getFrameworkName(const Framework& framework);

VS_DISCOVERY_EXPORT bool isFrameworkAvailable(const Framework& framework);

/**
 * @brief Returns framework backend with lower FrameworkStatus.
 *        If all backends have same level, returns prefered one
 * @return best framework
 */
VS_DISCOVERY_EXPORT Framework getBestFramework(const Framework& preferedFramework = Framework::CUDA);

}  // namespace Discovery
}  // namespace VideoStitch
