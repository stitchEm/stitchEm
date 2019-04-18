// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"
#include "libvideostitch/allocator.hpp"
#include "gpu/stream.hpp"
#include "gpu/surface.hpp"

namespace VideoStitch {
namespace GPU {

Status depthJointBilateralFilter(GPU::Surface& output, const GPU::Surface& input, const Core::SourceSurface& texture,
                                 GPU::Stream& stream);

}
}  // namespace VideoStitch
