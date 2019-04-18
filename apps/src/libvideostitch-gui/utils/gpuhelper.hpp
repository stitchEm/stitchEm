// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"

#include "libgpudiscovery/genericDeviceInfo.hpp"

namespace VideoStitch {
namespace GPU {

VS_GUI_EXPORT bool checkGPUFrameworkAvailable(VideoStitch::Discovery::Framework framework);
VS_GUI_EXPORT void showGPUInitializationError(int device, std::string const& error);
}  // namespace GPU
}  // namespace VideoStitch
