// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "../../gpuKernelDef.h"

// needed by OpenCL
#define gpu_fn __device__
#define lut_ptr
#define image2d_t cudaTextureObject_t

inline __device__ float4 read_texture_vs(read_only image2d_t image, float2 uv) {
  return tex2D<float4>(image, uv.x, uv.y);
}

inline __device__ float2 read_coord_vs(read_only image2d_t image, float2 uv) {
  return tex2D<float2>(image, uv.x, uv.y);
}

inline __device__ __host__ float3 get_xyz(float4 v) { return make_float3(v.x, v.y, v.z); }

inline __device__ __host__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

#define OutputRectCropper_isPanoPointVisible OutputRectCropper::isPanoPointVisible
#define OutputCircleCropper_isPanoPointVisible OutputCircleCropper::isPanoPointVisible

#define LinearPhotoCorrection_setup LinearPhotoCorrection::setup
#define LinearPhotoCorrection_corr LinearPhotoCorrection::corr
#define LinearPhotoCorrection_invCorr LinearPhotoCorrection::invCorr

#define GammaPhotoCorrection_setup GammaPhotoCorrection::setup
#define GammaPhotoCorrection_corr GammaPhotoCorrection::corr
#define GammaPhotoCorrection_invCorr GammaPhotoCorrection::invCorr

#define EmorPhotoCorrection_setup EmorPhotoCorrection::setup
#define EmorPhotoCorrection_corr EmorPhotoCorrection::corr
#define EmorPhotoCorrection_invCorr EmorPhotoCorrection::invCorr
