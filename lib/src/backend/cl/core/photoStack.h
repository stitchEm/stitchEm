// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef _PHOTOSTACK_H_
#define _PHOTOSTACK_H_

// linear
static inline const global float* PhotoCorrection_linear_setup(const global float* floatPtr) { return floatPtr; }
static inline float3 PhotoCorrection_linear_corr(float3 color, float photoParam, const global float* floatPtr) {
  return color;
}
static inline float3 PhotoCorrection_linear_invCorr(float3 color, float photoParam, const global float* floatPtr) {
  return color;
}

// gamma
static inline const global float* PhotoCorrection_gamma_setup(const global float* floatPtr) { return floatPtr; }
static inline float3 PhotoCorrection_gamma_corr(float3 color, float photoParam, const global float* floatPtr) {
  return pow(color / 255.0f, photoParam);
}
static inline float3 PhotoCorrection_gamma_invCorr(float3 color, float photoParam, const global float* floatPtr) {
  const float invGamma = 1.0f / photoParam;
  float3 f = {255.f, 255.f, 255.f};
  return pow(color, invGamma) * f;
}

/**
 * Lookup f (in [0;1] in a lookup table).
 */
static inline float PhotoCorrection_emor_lookup(float f, const global float* lookupTable) {
  // When f == 1.0, then we get:
  // f == 1023.0, i == 1023, x == 0.0, and i + 1 == 1024.
  // Therefore we must allocate 1025 floats and put something valid in lookupTable[1024]
  // (The value does not matter as long as it's not nan of inf, it's multiplied by 0.0.
  const float fclamp = clamp(f, 0.0f, 1.0f) * 1023.0f;
  const int index = convert_int_rtn(fclamp);
  const float interp = fclamp - convert_float(index);  // in [0;1]
  const float lookupIndex = lookupTable[index];
  const float lookupNextIndex = lookupTable[index + 1];
  return (1.0f - interp) * lookupIndex + interp * lookupNextIndex;
}

static inline const global float* PhotoCorrection_emor_setup(const global float* floatPtr) { return floatPtr; }
static inline float3 PhotoCorrection_emor_corr(float3 color, float photoParam, const global float* floatPtr) {
  floatPtr += 1024;
  const float3 fromTable = {PhotoCorrection_emor_lookup(color.x / 255.0f, floatPtr),
                            PhotoCorrection_emor_lookup(color.y / 255.0f, floatPtr),
                            PhotoCorrection_emor_lookup(color.z / 255.0f, floatPtr)};
  return fromTable;
}

static inline float3 PhotoCorrection_emor_invCorr(float3 color, float photoParam, const global float* floatPtr) {
  const float3 fromTable = {PhotoCorrection_emor_lookup(color.x, floatPtr),
                            PhotoCorrection_emor_lookup(color.y, floatPtr),
                            PhotoCorrection_emor_lookup(color.z, floatPtr)};
  return 255.0f * fromTable;
}

#endif
