// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Input {
Status overlayCheckerBoard(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, unsigned checkerSize,
                           uint32_t color1, uint32_t color2, uint32_t color3, GPU::Stream stream);
}
}  // namespace VideoStitch
