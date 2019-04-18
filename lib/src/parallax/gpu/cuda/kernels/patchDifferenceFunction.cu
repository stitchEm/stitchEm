// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/imageOps.hpp"

#include "parallax/flowConstant.hpp"

#ifndef VS_OPENCL
#include <cuda_runtime.h>
#endif
#include <stdint.h>
#include <vector_types.h>
#include <device_functions.h>
#include <vector_functions.h>

typedef float4 mat2;

namespace VideoStitch {

inline __device__ float length(int2 v) { return __fsqrt_rn(float(v.x * v.x + v.y * v.y)); }

inline __host__ __device__ float sqr(const float x) { return x * x; }

static __inline__ __host__ __device__ mat2 make_mat2(float a00, float a01, float a10, float a11) {
  mat2 t;
  t.x = a00;
  t.y = a01;
  t.z = a10;
  t.w = a11;
  return t;
}

inline __host__ __device__ mat2 transpose(mat2 t) { return make_mat2(t.x, t.z, t.y, t.w); }

inline __device__ __host__ mat2 operator*(mat2 a, mat2 b) {
  return make_mat2(a.x * b.x + a.y * b.z, a.x * b.y + a.y * b.w, a.z * b.x + a.w * b.z, a.z * b.y + a.w * b.w);
}

inline __device__ __host__ mat2 operator*(float v, mat2 a) { return make_mat2(a.x * v, a.y * v, a.z * v, a.w * v); }

inline __device__ __host__ float2 operator*(mat2 a, float2 v) {
  return make_float2(a.x * v.x + a.y * v.y, a.z * v.x + a.w * v.y);
}

inline __device__ __host__ int min(int a, int b) { return a < b ? a : b; }

inline __device__ __host__ int max(int a, int b) { return a > b ? a : b; }

inline __device__ __host__ float min(float a, float b) { return a < b ? a : b; }

inline __device__ __host__ float max(float a, float b) { return a > b ? a : b; }

inline __device__ __host__ int2 min(int2 a, int2 b) { return make_int2(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y); }

inline __device__ __host__ int2 max(int2 a, int2 b) { return make_int2(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y); }

namespace Core {

inline __host__ __device__ float getCost(const int windowSize, const float gradientWeight, const int2 size0,
                                         const uint32_t* input0, const float* gradient0, const int2 coord0,
                                         const int2 size1, const uint32_t* input1, const float* gradient1,
                                         const int2 coord1) {
  // Check if this is a valid pixel
  if (!inRange(coord1, size1)) {
    return MAX_INVALID_COST;
  }
  if (!inRange(coord0, size0)) {
    return MAX_INVALID_COST;
  }
  uint32_t v0 = input0[coord0.y * size0.x + coord0.x];
  if (Image::RGBA::a(v0) == 0) {
    return MAX_INVALID_COST;
  }
  uint32_t v1 = input1[coord1.y * size1.x + coord1.x];
  if (Image::RGBA::a(v1) == 0) {
    return MAX_INVALID_COST;
  }
  float cost = 0;
  float outOfBound = 0;
  for (int i = -windowSize; i <= windowSize; i++)
    for (int j = -windowSize; j <= windowSize; j++) {
      const int2 c0 = make_int2(coord0.x + i, coord0.y + j);
      const int2 c1 = make_int2(coord1.x + i, coord1.y + j);
      if (inRange(c0, size0) && inRange(c1, size1)) {
        uint32_t v0 = input0[c0.y * size0.x + c0.x];
        uint32_t v1 = input1[c1.y * size1.x + c1.x];
        if (Image::RGBA::a(v0) > 0 && Image::RGBA::a(v1) > 0) {
          float g0 = min(1.0f, gradient0[c0.y * size0.x + c0.x]);
          float g1 = min(1.0f, gradient1[c1.y * size1.x + c1.x]);
          const float sad = abs((float(Image::RGBA::r(v0)) - Image::RGBA::r(v1)) / 255.0f) +
                            abs((float(Image::RGBA::g(v0)) - Image::RGBA::g(v1)) / 255.0f) +
                            abs((float(Image::RGBA::b(v0)) - Image::RGBA::b(v1)) / 255.0f) +
                            gradientWeight * abs(g0 - g1);
          cost += sad;
        } else {
          outOfBound++;
        }
      } else {
        outOfBound++;
      }
    }
  cost += outOfBound * 1.0f;
  return cost / float((windowSize * 2 + 1) * (windowSize * 2 + 1));
}

inline __host__ __device__ float2 getDualCost(const int windowSize, const float gradientWeight, const int2 size0,
                                              const uint32_t* input0, const float* gradient0, const int2 coord0,
                                              const int2 size1, const uint32_t* input1, const float* gradient1,
                                              const int2 coord1A, const int2 coord1B) {
  // Check if this is a valid pixel
  float2 cost = make_float2(0, 0);
  float2 outOfBound = make_float2(0, 0);

  // Check if this is a valid pixel
  if (!inRange(coord1A, size1)) {
    cost.x = MAX_INVALID_COST;
  }
  if (!inRange(coord1B, size1)) {
    cost.y = MAX_INVALID_COST;
  }
  if (cost.x == MAX_INVALID_COST && cost.y == MAX_INVALID_COST) {
    return cost;
  }

  uint32_t vA = input0[coord1A.y * size1.x + coord1A.x];
  uint32_t vB = input0[coord1B.y * size1.x + coord1B.x];
  if (Image::RGBA::a(vA) == 0) {
    cost.x = MAX_INVALID_COST;
  }
  if (Image::RGBA::a(vB) == 0) {
    cost.y = MAX_INVALID_COST;
  }
  if (cost.x == MAX_INVALID_COST && cost.y == MAX_INVALID_COST) {
    return cost;
  }

  for (int i = -windowSize; i <= windowSize; i++)
    for (int j = -windowSize; j <= windowSize; j++) {
      const int2 c0 = make_int2(coord0.x + i, coord0.y + j);

      const int2 c1A = make_int2(coord1A.x + i, coord1A.y + j);
      const int2 c1B = make_int2(coord1B.x + i, coord1B.y + j);
      if (inRange(c0, size0)) {
        uint32_t v0 = input0[c0.y * size0.x + c0.x];
        float g0 = gradient0[c0.y * size0.x + c0.x];

        if (cost.x != MAX_INVALID_COST && inRange(c1A, size1)) {
          uint32_t v1 = input1[c1A.y * size1.x + c1A.x];
          if (Image::RGBA::a(v0) > 0 && Image::RGBA::a(v1) > 0) {
            float g1 = gradient1[c1A.y * size1.x + c1A.x];
            const float sadA = abs((float(Image::RGBA::r(v0)) - Image::RGBA::r(v1)) / 255.0f) +
                               abs((float(Image::RGBA::g(v0)) - Image::RGBA::g(v1)) / 255.0f) +
                               abs((float(Image::RGBA::b(v0)) - Image::RGBA::b(v1)) / 255.0f) +
                               gradientWeight * abs(g0 - g1);
            cost.x += sadA;
          } else {
            outOfBound.x++;
          }
        }

        if (cost.y != MAX_INVALID_COST && inRange(c1B, size1)) {
          uint32_t v1 = input1[c1B.y * size1.x + c1B.x];
          if (Image::RGBA::a(v0) > 0 && Image::RGBA::a(v1) > 0) {
            float g1 = gradient1[c1B.y * size1.x + c1B.x];
            const float sadB = abs((float(Image::RGBA::r(v0)) - Image::RGBA::r(v1)) / 255.0f) +
                               abs((float(Image::RGBA::g(v0)) - Image::RGBA::g(v1)) / 255.0f) +
                               abs((float(Image::RGBA::b(v0)) - Image::RGBA::b(v1)) / 255.0f) +
                               gradientWeight * abs(g0 - g1);
            cost.y += sadB;
          } else {
            outOfBound.y++;
          }
        }

      } else {
        outOfBound.x++;
        outOfBound.y++;
      }
    }
  if (cost.x != MAX_INVALID_COST) {
    cost.x = (cost.x + outOfBound.x) / (windowSize * 2 + 1) * (windowSize * 2 + 1);
  }
  if (cost.y != MAX_INVALID_COST) {
    cost.y = (cost.y + outOfBound.y) / (windowSize * 2 + 1) * (windowSize * 2 + 1);
  }
  return cost;
}

inline __host__ __device__ float getCUR(const int windowSize, const float gradientWeight, const int2 size0,
                                        const uint32_t* input0, const float* gradient0, const int2 coord0,
                                        const int2 size1, const uint32_t* input1, const float* gradient1,
                                        const int2 coord1) {
  float c11 = getCost(windowSize, gradientWeight, size0, input0, gradient0, coord0, size1, input1, gradient1, coord1);
  float c21 = (coord1.x + 1 < size1.x) ? getCost(windowSize, gradientWeight, size0, input0, gradient0, coord0, size1,
                                                 input1, gradient1, make_int2(coord1.x + 1, coord1.y))
                                       : c11;
  if (c21 == MAX_INVALID_COST) c21 = c11;
  float c01 = (coord1.x - 1 >= 0) ? getCost(windowSize, gradientWeight, size0, input0, gradient0, coord0, size1, input1,
                                            gradient1, make_int2(coord1.x - 1, coord1.y))
                                  : c11;
  if (c01 == MAX_INVALID_COST) c01 = c11;
  float c12 = (coord1.y + 1 < size1.y) ? getCost(windowSize, gradientWeight, size0, input0, gradient0, coord0, size1,
                                                 input1, gradient1, make_int2(coord1.x, coord1.y + 1))
                                       : c11;
  if (c12 == MAX_INVALID_COST) c12 = c11;
  float c10 = (coord1.y - 1 >= 0) ? getCost(windowSize, gradientWeight, size0, input0, gradient0, coord0, size1, input1,
                                            gradient1, make_int2(coord1.x, coord1.y - 1))
                                  : c11;
  if (c10 == MAX_INVALID_COST) c10 = c11;
  return (c21 + c01 + c12 + c10 - 4 * c11);
}

}  // namespace Core
}  // namespace VideoStitch
