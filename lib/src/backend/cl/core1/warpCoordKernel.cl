// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "warpKernelDef.h"

static inline void coord_write(float2 c, write_only image2d_t img, int x, int y) {
  write_imagef(img, (int2){x, y}, (float4)(c, c));
}

#include "backend/common/core1/warpCoordKernel.gpu"
