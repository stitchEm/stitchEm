// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/core/transformPhotoParam.hpp"

#ifdef VS_OPENCL
#define CUDART_PI_F 3.141592654f

#define __device__
#define __host__

#endif

#ifdef __CUDA_ARCH__
#define powf_photo_vs __powf
#define saturatef_vs __saturatef
#define float2int_rd_vs __float2int_rd
#else
#define powf_photo_vs powf
#define saturatef_vs(f) (f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f))
#define float2int_rd_vs (int)
#endif

#include <cmath>

namespace VideoStitch {
namespace Core {

/**
 * Photometric response correction functions:
 */

struct LinearPhotoCorrection {
  // enum SHMem { SharedMemoryNeed = 0 };
  static inline __device__ __host__ const float* setup(const float* floatPtr) { return floatPtr; }
  static inline __device__ __host__ float3 corr(float3 c, float /* photoParam */, const float* /* floatPtr */) {
    return c;
  }
  static inline __device__ __host__ float3 invCorr(float3 c, float /* photoParam */, const float* /* floatPtr */) {
    return c;
  }
};

struct GammaPhotoCorrection {
  // enum SHMem { SharedMemoryNeed = 0 };
  static inline __device__ __host__ const float* setup(const float* floatPtr) { return floatPtr; }
  static inline __device__ __host__ float3 corr(float3 color, float gamma, const float* /* floatPtr */) {
    color.x = powf_photo_vs(color.x / 255.0f, gamma);
    color.y = powf_photo_vs(color.y / 255.0f, gamma);
    color.z = powf_photo_vs(color.z / 255.0f, gamma);
    return color;
  }

  static inline __device__ __host__ float3 invCorr(float3 color, float gamma, const float* /* floatPtr */) {
    const float invGamma = 1.0f / gamma;
    color.x = 255.0f * powf_photo_vs(color.x, invGamma);
    color.y = 255.0f * powf_photo_vs(color.y, invGamma);
    color.z = 255.0f * powf_photo_vs(color.z, invGamma);
    return color;
  }
};

struct EmorPhotoCorrection {
  // The parameter is a lookup table of size 1024 * 2 (direct then inverse)
  static inline __device__ __host__ const float* setup(const float* floatPtr) {
#ifdef __CUDA_ARCH__
    __shared__ float lut[2049];
    const unsigned threadId = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = threadId; i < 2049; i += blockDim.x * blockDim.y) {
      lut[i] = floatPtr[i];
    }
    __syncthreads();
    return lut;
#else
    return floatPtr;
#endif
  }

  /**
   * Lookup f (in [0;1] in a lookup table).
   */
  static inline __device__ __host__ float lookup(float f, const float* lookupTable) {
    // When f == 1.0, then we get:
    // f == 1023.0, i == 1023, x == 0.0, and i + 1 == 1024.
    // Therefore we must allocate 1025 floats and put something valid in lookupTable[1024]
    // (The value does not matter as long as it's not nan of inf, it's multiplied by 0.0.
    f = saturatef_vs(f) * 1023.0f;
    const int i = float2int_rd_vs(f);
    const float x = f - i;  // in [0;1]
    return (1.0f - x) * lookupTable[i] + x * lookupTable[i + 1];
  }

  static inline __device__ __host__ float3 corr(float3 color, float /* floatParam */, const float* lookupTable) {
    const float* floatPtr = lookupTable;
    floatPtr += 1024;
    color.x = lookup(color.x / 255.0f, floatPtr);
    color.y = lookup(color.y / 255.0f, floatPtr);
    color.z = lookup(color.z / 255.0f, floatPtr);
    return color;
  }

  static inline __device__ __host__ float3 invCorr(float3 color, float /* floatParam */, const float* lookupTable) {
    const float* floatPtr = lookupTable;
    color.x = 255.0f * lookup(color.x, floatPtr);
    color.y = 255.0f * lookup(color.y, floatPtr);
    color.z = 255.0f * lookup(color.z, floatPtr);
    return color;
  }
};
}  // namespace Core
}  // namespace VideoStitch

#undef powf_photo_vs
#undef saturatef_vs
#undef float2int_rd_vs
