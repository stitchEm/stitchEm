// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma OPENCL FP_CONTRACT ON

#define INL_FUNCTION_NAME(fnName) fnName
#define INL_TRANSFORM_FN_1D(fnName) static inline float fnName

#define INL_CONVERT_3D_2D_FN(fnName) static inline float2 fnName
#define INL_CONVERT_2D_3D_FN(fnName) static inline float3 fnName

#define INL_PERSPECTIVE_TRANSFORM_FN(fnName) static inline float3 fnName
#define INL_ROTATION_TRANSFORM_FN(fnName) static inline float3 fnName
#define INL_DISTORTION_TRANSFORM_FN(fnName) static inline float2 fnName

#define length_vs fast_length

static inline float invLength2(float2 v) { return native_rsqrt(v.x * v.x + v.y * v.y); }

static inline float invLength3(float3 v) { return native_rsqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

#define acosf_vs acos
#define asinf_vs asin
#define atan2f_vs atan2
#define clampf_vs clamp
#define cosf_vs cos
#define fabsf_vs fabs
#define powf_vs pow
#define sinf_vs sin
#define sqrtf_vs sqrt
#define tanf_vs tan

#define acos_vs acos
#define cos_vs cos
#define fabs_vs fabs
#define pow_vs pow
#define sqrt_vs sqrt

#define solve_float_t float
#define solve_float_t2 float2
#define solve_float_t3 float3
#define make_solve_float_t2 make_float2
#define make_solve_float_t3 make_float3

#define solveQuartic_vs solveQuartic

#ifndef make_float2
#define make_float2(A, B) (float2)((A), (B))
#endif  // make_float2

#ifndef make_float3
#define make_float3(A, B, C) (float3)((A), (B), (C))
#endif  // make_float3

#ifndef make_float4
#define make_float4(A, B, C, D) (float4)((A), (B), (C), (D))
#endif  // make_float4

#define PI_F_VS M_PI_F

#undef FN_NAME
#undef INL_FN_FLOAT
#undef INL_FN_FLOAT2
#undef INL_FN_FLOAT3
#undef INL_FN_QUARTIC_SOL

#define FN_NAME(fnName) fnName
#define INL_FN_FLOAT(fnName) static inline solve_float_t fnName
#define INL_FN_FLOAT2(fnName) static inline solve_float_t2 fnName
#define INL_FN_FLOAT3(fnName) static inline solve_float_t3 fnName
#define INL_FN_QUARTIC_SOL(fnName) static inline vsQuarticSolution fnName

#define convert_float (float)

#include "backend/common/core/quarticSolver.gpu"
#include "backend/common/core/transformStack.gpu"

#undef INL_TRANSFORM_FN_1D
#undef INL_CONVERT_3D_2D_FN
#undef INL_CONVERT_2D_3D_FN
#undef INL_PERSPECTIVE_TRANSFORM_FN
#undef INL_ROTATION_TRANSFORM_FN
#undef INL_DISTORTION_TRANSFORM_FN
