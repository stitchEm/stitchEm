// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/kernels/photoStack.cu"
#include "core/kernels/withinStack.cu"
#include "mapFunction.cu"

namespace VideoStitch {
namespace Core {
#include "distortionKernel.gpu"
}
}  // namespace VideoStitch
