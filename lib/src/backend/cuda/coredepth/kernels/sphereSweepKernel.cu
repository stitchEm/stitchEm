// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

// TODODEPTH are all these includes needed?

#include "backend/cuda/parallax/kernels/mapInverseFunction.cu"

#include "backend/common/imageOps.hpp"
#include "core/kernels/withinStack.cu"
#include "backend/cuda/core1/kernels/mapFunction.cu"
#include "backend/cuda/core1/kernels/defKernel.cu"
#include "backend/cuda/surface.hpp"
// #include "backend/cuda/core1/warpMergerKernelDef.h"

#include "backend/common/vectorOps.hpp"

// TODODEPTH move these to vectorOps.hpp

inline __device__ __host__ float4 operator/(float4 v, float a) {
  return make_float4(v.x / a, v.y / a, v.z / a, v.w / a);
}

inline __device__ __host__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ __host__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__inline__ __device__ __host__ float4 fmaxf(float4 a, float4 b) {
  return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline __device__ __host__ float4 sqrt4(float4 val) {
  return make_float4(sqrtf(val.x), sqrtf(val.y), sqrtf(val.z), sqrtf(val.w));
}

inline __device__ __host__ float4 log4(float4 val) {
  return make_float4(logf(val.x), logf(val.y), logf(val.z), logf(val.w));
}

inline __device__ __host__ float4 exp4(float4 val) {
  return make_float4(expf(val.x), expf(val.y), expf(val.z), expf(val.w));
}

inline __device__ __host__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline __device__ __host__ int2 convert_int2(float2 val) { return make_int2((int)val.x, (int)val.y); }

inline __device__ float read_depth_vs(surface_t surf, float2 uv) {
  float val;
  surf2Dread(&val, surf, uv.x * sizeof(float), uv.y);
  return val;
}

#define sync_threads __syncthreads
#define CLK_GLOBAL_MEM_FENCE
#define __globalmem__

#define INPUT_PARAMS_T const struct InputParams6

inline __device__ struct InputParams6 get_input_params(INPUT_PARAMS_T in) { return in; }

namespace VideoStitch {
namespace Core {

#include "backend/common/coredepth/sphereSweep.gpu"

}
}  // namespace VideoStitch
