// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/cl/gpuKernelDef.h"
#include "backend/cl/image/imageFormat.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

#include <backend/common/image/imageOps.gpu>

#pragma clang diagnostic pop
