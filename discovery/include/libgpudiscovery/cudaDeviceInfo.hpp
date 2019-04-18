// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "genericDeviceInfo.hpp"

#include <vector>

namespace VideoStitch {
namespace Discovery {

// returns true and fills the 2nd argument if a CUDA device corresponds to the vsDeviceIndex. Otherwise, it returns
// false and doesn't fill the 2nd argument
VS_DISCOVERY_EXPORT bool getCudaDeviceProperties(unsigned vsDeviceIndex, struct DeviceProperties&);

}  // namespace Discovery
}  // namespace VideoStitch
