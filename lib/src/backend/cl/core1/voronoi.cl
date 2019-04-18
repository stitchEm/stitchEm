// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../gpuKernelDef.h"

#include "../core/transformStack.cl"

#include "../../../core/panoDimensionsDef.hpp"

static inline float compute_distSqr(int x1, int y1, int x2, int y2, PanoRegion region_unused) {
  int2 p1 = {x1, y1};
  int2 p2 = {x2, y2};
  return length(convert_float2(p1 - p2));
}

#define length_vs fast_length

#define sync_threads barrier

#include "../../common/core1/voronoi.gpu"
