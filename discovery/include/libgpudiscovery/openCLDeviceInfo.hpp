// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "genericDeviceInfo.hpp"
#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif  //__APPLE

namespace VideoStitch {
namespace Discovery {

struct OpenCLDevice {
  struct DeviceProperties gpuProperties;
  size_t max_image_height;
  cl_platform_id platform_id;
  cl_device_id device_id;
};

// returns true and fills the 2nd argument if an OpenCL device corresponds to the vsDeviceIndex.
// Otherwise, it returns false and doesn't fill the 2nd argument
VS_DISCOVERY_EXPORT bool getOpenCLDeviceProperties(unsigned vsDeviceIndex, OpenCLDevice&);

}  // namespace Discovery
}  // namespace VideoStitch
