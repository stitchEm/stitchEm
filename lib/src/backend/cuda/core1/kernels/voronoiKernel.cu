// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef _VORONOIKERNEL_H_
#define _VORONOIKERNEL_H_

#include "../../core/transformStack.cu"

#include "../../gpuKernelDef.h"

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

#include <float.h>

namespace VideoStitch {
namespace Core {

#define sqrDist Image::sqrDist

#define length_vs length

#define sync_threads __syncthreads
#define CLK_GLOBAL_MEM_FENCE

__device__ float compute_distSqr(int x1, int y1, int x2, int y2, PanoRegion /* region */) {
  return sqrt((float)sqrDist(x1, y1, x2, y2));
}

#include "../../../common/core1/voronoi.gpu"

}  // namespace Core
}  // namespace VideoStitch

#endif  // _VORONOIKERNEL_H_
