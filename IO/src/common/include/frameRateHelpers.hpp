// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/frame.hpp"

namespace VideoStitch {
namespace Util {
/**
 * @brief Converts the fps value into FrameRate (num, den).
 * @param fps The fps value.
 * @return The FrameRate
 */
FrameRate fpsToNumDen(const double fps);
}  // namespace Util
}  // namespace VideoStitch
