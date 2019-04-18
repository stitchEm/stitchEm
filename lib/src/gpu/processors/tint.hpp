// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {
Status tint(GPU::Surface& dst, unsigned width, unsigned height, int32_t r, int32_t g, int32_t b, GPU::Stream stream);
}
}  // namespace VideoStitch
