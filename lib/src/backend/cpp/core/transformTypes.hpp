// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TRANSFORMTYPES_HPP_
#define TRANSFORMTYPES_HPP_

#include "gpu/vectorTypes.hpp"

namespace VideoStitch {
namespace Core {

#include "backend/common/core/transformPhotoParam.hpp"
#include "backend/common/core/types.hpp"

typedef float2 (*Convert3D2DFnT)(const float3);
typedef float3 (*Convert2D3DFnT)(const float2);
typedef float2 (*TransformFnT)(float2);
typedef float2 (*ParametrizedTransformFnT)(float2, float);
typedef float3 (*LiftTransformFnT)(float3, const float2);
typedef float2 (*ViewpointTranslationFnT)(float2, const float, const float, const float);
typedef float3 (*PerspectiveTransformFnt)(float3, const vsfloat3x4);
typedef float3 (*RotationTransformFnt)(float3, const vsfloat3x3);
typedef float2 (*DistortionTransformFnt)(float2, const vsDistortion);
typedef bool (*IsWithinFnT)(float2, float, float, float, float, float, float);
typedef void (*PhotoCorrectionFnT)(float&, float&, float&, struct TransformPhotoParam);

}  // namespace Core
}  // namespace VideoStitch

#endif
