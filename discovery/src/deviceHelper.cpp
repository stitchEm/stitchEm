// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "deviceHelper.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace VideoStitch {
namespace Discovery {

Vendor detectDeviceVendor(std::string deviceVendor) {
  std::transform(deviceVendor.begin(), deviceVendor.end(), deviceVendor.begin(), ::toupper);
  if ((deviceVendor.find("AMD") != std::string::npos) ||
      (deviceVendor.find("ADVANCED MICRO DEVICES") != std::string::npos)) {
    return Vendor::AMD;
  }
  if (deviceVendor.find("INTEL") != std::string::npos) {
    return Vendor::INTEL;
  }
  if (deviceVendor.find("NVIDIA") != std::string::npos) {
    return Vendor::NVIDIA;
  }
  return Vendor::UNKNOWN;
}

/**
 * @brief Erases all the occurrences of src from array.
 * @param array The array to be modified.
 * @param src String to be removed.
 */
void eraseStr(char array[], const std::string& src) {
  std::string converted(array);
  auto id = converted.find(src);
  while (id != std::string::npos) {
    converted.erase(id, src.length());
    id = converted.find(src);
  }
  memcpy(array, converted.c_str(), converted.length() + 1);
}

/**
 * @brief Prepends a string to a DeviceProperties name, if it fits
 * @param prop The device properties name to be modified
 * @param name The string to prepend.
 */
void prependName(Discovery::DeviceProperties& prop, const std::string& name) {
  std::string converted(prop.name);
  converted = name + " " + converted;
  if (converted.length() < sizeof prop.name) {
    memcpy(prop.name, converted.c_str(), converted.length() + 1);
  }
}

/**
 * @brief Replaces / removes elements from the device name provided by OpenCL.
 * @param prop The device property.
 */
void improveDeviceName(Discovery::DeviceProperties& prop) {
  // Improve the device name provided by OpenCL
  eraseStr(prop.name, "(R)");
  eraseStr(prop.name, "Compute Engine");
  eraseStr(prop.name, "(TM)");

  std::string name(prop.name);

  Vendor vendorInDeviceName = detectDeviceVendor(name);

  if (vendorInDeviceName == Vendor::UNKNOWN && prop.vendor != Vendor::UNKNOWN) {
    std::string vendorString;
    switch (prop.vendor) {
      case Vendor::AMD:
        vendorString = "AMD";
        break;
      case Vendor::INTEL:
        vendorString = "Intel";
        break;
      case Vendor::NVIDIA:
        vendorString = "Nvidia";
        break;
      case Vendor::UNKNOWN:
        // impossible
        assert(false);
        break;
    }

    prependName(prop, vendorString);
  }
}

}  // namespace Discovery
}  // namespace VideoStitch
