// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

Status callDummyKernel(GPU::Buffer<float> outputBuff, const GPU::Buffer<const float>& inputBuff,
                       unsigned int nbElements, float mult, GPU::Stream stream);

}
}  // namespace VideoStitch
