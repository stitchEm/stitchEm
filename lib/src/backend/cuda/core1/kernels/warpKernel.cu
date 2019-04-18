// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/imageOps.hpp"
#include "core/kernels/withinStack.cu"
#include "mapFunction.cu"
#include "defKernel.cu"
#include "backend/cuda/surface.hpp"
#include "backend/cuda/core1/warpMergerKernelDef.h"

namespace VideoStitch {
namespace Core {

#include "backend/common/core1/cubemapMapKernel.gpu"
#include "backend/common/core1/warpCoordInputKernel.gpu"
#include "backend/common/core1/warpCoordKernel.gpu"
#include "backend/common/core1/warpLookupKernel.gpu"
#include "backend/common/core1/warpKernel.gpu"

#include "warpFaceKernel.cu"

}  // namespace Core
}  // namespace VideoStitch
