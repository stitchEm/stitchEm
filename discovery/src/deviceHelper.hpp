// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libgpudiscovery/genericDeviceInfo.hpp"

#include <algorithm>

namespace VideoStitch {
namespace Discovery {

Vendor detectDeviceVendor(std::string deviceVendor);

/**
 * @brief Replaces / removes elements from the device name provided by OpenCL.
 * @param prop The device property.
 */
void improveDeviceName(Discovery::DeviceProperties& prop);

}  // namespace Discovery
}  // namespace VideoStitch
