// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"
#include <stdint.h>

namespace VideoStitch {
namespace Core {

Status grid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth, uint32_t color,
            uint32_t bgColor, GPU::Stream stream);

Status transparentForegroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t bgColor, GPU::Stream stream);

Status transparentBackgroundGrid(GPU::Buffer<uint32_t> dst, unsigned width, unsigned height, int size, int lineWidth,
                                 uint32_t color, GPU::Stream stream);
}  // namespace Core
}  // namespace VideoStitch
