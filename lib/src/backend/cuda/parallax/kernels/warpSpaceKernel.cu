// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/kernels/withinStack.cu"
#include "parallax/flowConstant.hpp"

namespace VideoStitch {
namespace Core {

#include "backend/common/parallax/warpCoordInputToOutputKernel.gpu"
#include "backend/common/parallax/warpCoordInputToOutputKernelSphere.gpu"
#include "backend/common/parallax/warpCoordOutputToInputKernel.gpu"
#include "backend/common/parallax/warpCoordOutputToInputKernelSphere.gpu"

}  // namespace Core
}  // namespace VideoStitch
