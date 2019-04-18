// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "transformTypes.hpp"

#include <cmath>
#include <cfloat>

// Defines depending on the environment.
#include "backend/common/vectorOps.hpp"
#include "core/quarticSolver.hpp"

#define INL_FUNCTION_NAME(fnName) fnName
#define INL_TRANSFORM_FN_1D(fnName) inline float fnName
#define INL_CONVERT_3D_2D_FN(fnName) inline float2 fnName
#define INL_CONVERT_2D_3D_FN(fnName) inline float3 fnName
#define INL_PERSPECTIVE_TRANSFORM_FN(fnName) inline float3 fnName
#define INL_ROTATION_TRANSFORM_FN(fnName) inline float3 fnName
#define INL_DISTORTION_TRANSFORM_FN(fnName) inline float2 fnName

#define acosf_vs acosf
#define asinf_vs asinf
#define atan2f_vs atan2f
#define cosf_vs cosf
#define fabsf_vs fabsf
#define invLength2 invLength
#define invLength3 invLength
#define length_vs length
#define powf_vs powf
#define sinf_vs sinf
#define sqrtf_vs sqrtf
#define tanf_vs tanf

#define clampf_vs clampf

#define solveQuartic_vs Core::TransformStack::solveQuartic
#define PI_F_VS CUDART_PI_F

#define dummyParamName /* dummyParamName */

#define convert_float static_cast<float>

#define acos_vs acos
#define cos_vs cos
#define fabs_vs fabs
#define pow_vs pow
#define sqrt_vs sqrt

#define solve_float_t double
#define solve_float_t2 double2
#define solve_float_t3 double3

#define make_solve_float_t2 make_double2
#define make_solve_float_t3 make_double3

namespace VideoStitch {
namespace Core {
namespace TransformStack {

#include "backend/common/core/quarticSolver.gpu"
#include "backend/common/core/transformStack.gpu"

}  // namespace TransformStack
}  // namespace Core
}  // namespace VideoStitch

#undef INL_FUNCTION_NAME
#undef INL_TRANSFORM_FN_1D
#undef INL_CONVERT_3D_2D_FN
#undef INL_CONVERT_2D_3D_FN
#undef INL_PERSPECTIVE_TRANSFORM_FN
#undef INL_ROTATION_TRANSFORM_FN
#undef INL_DISTORTION_TRANSFORM_FN

#undef acosf_vs
#undef asinf_vs
#undef atan2f_vs
#undef cosf_vs
#undef fabsf_vs
#undef invLength2
#undef invLength3
#undef length_vs
#undef powf_vs
#undef sinf_vs
#undef sqrtf_vs
#undef tanf_vs

#undef clampf_vs

#undef solveQuartic_vs
#undef PI_F_VS

#undef dummyParamName

#undef convert_float

#undef acos_vs
#undef cos_vs
#undef fabs_vs
#undef pow_vs
#undef sqrt_vs

#undef solve_float_t
#undef solve_float_t2
#undef solve_float_t3

#undef make_solve_float_t2
#undef make_solve_float_t3
