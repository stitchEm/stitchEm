// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <stdint.h>
#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace Output {

Status anaglyphColorLeft(uint32_t* dst, const uint32_t* src, const int64_t height, const int64_t width);

Status anaglyphColorRight(uint32_t* dst, const uint32_t* src, const int64_t height, const int64_t width);

}  // namespace Output
}  // namespace VideoStitch
