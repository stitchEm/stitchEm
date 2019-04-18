// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"
#include <stdint.h>

namespace VideoStitch {
namespace Input {
Status maskInput(GPU::Surface& dst, GPU::Buffer<const unsigned char> maskDevBufferP, unsigned width, unsigned height,
                 GPU::Stream stream);
}
}  // namespace VideoStitch
