// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/surface.hpp"
#include "gpu/stream.hpp"

//# include <stdint.h>

namespace VideoStitch {
namespace Core {
Status maskOverlay(GPU::Surface& dst, unsigned width, unsigned height, uint32_t color, GPU::Stream stream);
}
}  // namespace VideoStitch
