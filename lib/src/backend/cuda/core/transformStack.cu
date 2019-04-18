// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/vectorOps.hpp"

#include <math_constants.h>
#include <cfloat>

#define INL_FUNCTION_NAME(fnName) fnName
#define INL_TRANSFORM_FN_1D(fnName) inline __device__ float fnName
#define INL_CONVERT_3D_2D_FN(fnName) inline __device__ float2 fnName
#define INL_CONVERT_2D_3D_FN(fnName) inline __device__ float3 fnName
#define INL_PERSPECTIVE_TRANSFORM_FN(fnName) inline __device__ float3 fnName
#define INL_ROTATION_TRANSFORM_FN(fnName) inline __device__ float3 fnName
#define INL_DISTORTION_TRANSFORM_FN(fnName) inline __device__ float2 fnName

#undef FN_NAME
#undef INL_FN_FLOAT
#undef INL_FN_FLOAT2
#undef INL_FN_FLOAT3
#undef INL_FN_QUARTIC_SOL

#define FN_NAME(fnName) fnName
#define INL_FN_FLOAT(fnName) inline __device__ double fnName
#define INL_FN_FLOAT2(fnName) inline __device__ double2 fnName
#define INL_FN_FLOAT3(fnName) inline __device__ double3 fnName
#define INL_FN_QUARTIC_SOL(fnName) inline __device__ vsQuarticSolution fnName

#define length_vs length

#define fabsf_vs fabsf
#define asinf_vs asinf
#define acosf_vs acosf
#define cosf_vs cosf
#define sinf_vs sinf
#define tanf_vs tanf
#define atanf_vs atanf
#define atan2f_vs atan2f
#define sqrtf_vs sqrtf
#define powf_vs powf
#define invLength2 invLength
#define invLength3 invLength

#define acos_vs acos
#define cos_vs cos
#define fabs_vs fabs
#define pow_vs pow
#define sqrt_vs sqrt

#define solve_float_t double
#define solve_float_t2 double2
#define solve_float_t3 double3

#define convert_float static_cast<float>

#define make_solve_float_t2 make_double2
#define make_solve_float_t3 make_double3

#define solveQuartic_vs Core::solveQuartic

#define clampf_vs clampf

#define PI_F_VS CUDART_PI_F

#define dummyParamName /* dummyParamName */

namespace VideoStitch {
namespace Core {
#include "backend/common/core/quarticSolver.gpu"
#include "backend/common/core/transformStack.gpu"
}  // namespace Core
}  // namespace VideoStitch

#undef INL_TRANSFORM_FN_1D
#undef INL_CONVERT_3D_2D_FN
#undef INL_CONVERT_2D_3D_FN
#undef INL_PERSPECTIVE_TRANSFORM_FN
#undef INL_ROTATION_TRANSFORM_FN
#undef INL_DISTORTION_TRANSFORM_FN
