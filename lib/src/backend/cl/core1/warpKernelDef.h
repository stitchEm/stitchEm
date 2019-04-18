// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../gpuKernelDef.h"

#define lut_ptr global_mem
static __constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

static inline float4 read_texture_vs(read_only image2d_t tex, float2 uv) { return read_imagef(tex, smp, uv); }

static inline float2 read_coord_vs(read_only image2d_t tex, float2 uv) { return read_imagef(tex, smp, uv).xy; }

static inline void write_texture_vs(float4 color, write_only image2d_t img, int2 coords) {
  write_imagef(img, coords, color);
}

static inline void surface_read_depth(float *depth, surface_t surf, int x, int y) {
  // TODO FIXME
  // no way to write a float to a 2D surface in OpenCL yet
}

static inline void surface_write_depth(float depth, surface_t surf, int x, int y) {
  // TODO FIXME
  // no way to write a float to a 2D surface in OpenCL yet
}

#include "../core/photoStack.h"

#define LinearPhotoCorrection_setup PhotoCorrection_linear_setup
#define LinearPhotoCorrection_corr PhotoCorrection_linear_corr
#define LinearPhotoCorrection_invCorr PhotoCorrection_linear_invCorr

#define GammaPhotoCorrection_setup PhotoCorrection_gamma_setup
#define GammaPhotoCorrection_corr PhotoCorrection_gamma_corr
#define GammaPhotoCorrection_invCorr PhotoCorrection_gamma_invCorr

#define EmorPhotoCorrection_setup PhotoCorrection_emor_setup
#define EmorPhotoCorrection_corr PhotoCorrection_emor_corr
#define EmorPhotoCorrection_invCorr PhotoCorrection_emor_invCorr

static inline bool OutputRectCropper_isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) { return true; }

static inline bool OutputCircleCropper_isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) { return true; }

static inline bool isWithinCropRect(const float2 uv, float width, float height, float cLeft, float cRight, float cTop,
                                    float cBottom) {
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height && cLeft <= uv.x && uv.x <= cRight &&
         cTop <= uv.y && uv.y <= cBottom;
}

static inline bool isWithinCropCircle(const float2 uv, float width, float height, float cLeft, float cRight, float cTop,
                                      float cBottom) {
  const float centerX = (cRight + cLeft) / 2.0f;
  const float centerY = (cBottom + cTop) / 2.0f;
  const float radius = fmin(cRight - cLeft, cBottom - cTop) / 2.0f;
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height &&
         (uv.x - centerX) * (uv.x - centerX) + (uv.y - centerY) * (uv.y - centerY) < radius * radius;
}

static inline float3 get_xyz(float4 val) { return val.xyz; }

#define __float2int_rn convert_int_rte

#ifndef make_float2
#define make_float2(A, B) (float2)((A), (B))
#endif  // make_float2

#include "../image/imageFormat.h"

#include "backend/common/core/types.hpp"
#include "backend/common/core/transformPhotoParam.hpp"
#include "warpMergerKernelDef.h"

#include "mapFunction.h"
#include "photoCorrectionFunction.h"
