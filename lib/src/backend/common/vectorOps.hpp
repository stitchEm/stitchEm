// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/vectorTypes.hpp>

#include <stdint.h>
#include <math.h>

#ifdef VS_OPENCL
#define CUDART_PI_F 3.141592654f

#define __device__
#define __host__
#define __restrict__

#else
#include <math_constants.h>
#endif

inline __device__ __host__ float degToRad(float v) { return CUDART_PI_F * (v / 180.0f); }

inline __device__ __host__ float radToDeg(float v) { return v * (180.0f / CUDART_PI_F); }

inline __device__ __host__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }

inline __device__ __host__ int2 operator+(int2 a, int2 b) { return make_int2(a.x + b.x, a.y + b.y); }

inline __device__ __host__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ __host__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }

inline __device__ __host__ int2 operator-(int2 a, int2 b) { return make_int2(a.x - b.x, a.y - b.y); }

inline __device__ __host__ float2 operator*(float a, float2 v) { return make_float2(a * v.x, a * v.y); }

inline __device__ __host__ float3 operator*(float a, float3 v) { return make_float3(a * v.x, a * v.y, a * v.z); }

inline __device__ __host__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline __device__ __host__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

inline __device__ __host__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __device__ __host__ void operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __device__ __host__ void operator-=(float2 &a, float2 b) {
  a.x -= b.x;
  a.y -= b.y;
}

inline __device__ __host__ void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

inline __device__ __host__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline __device__ __host__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ __host__ void operator*=(float2 &v, float a) {
  v.x *= a;
  v.y *= a;
}

inline __device__ __host__ void operator*=(float2 &v, float2 a) {
  v.x *= a.x;
  v.y *= a.y;
}

inline __device__ __host__ void operator*=(float3 &v, float a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
}

inline __device__ __host__ void operator/=(float2 &v, float a) {
  v.x /= a;
  v.y /= a;
}

inline __device__ __host__ void operator/=(float2 &v, float2 a) {
  v.x /= a.x;
  v.y /= a.y;
}

inline __device__ __host__ void operator/=(float3 &v, float a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
}

inline __device__ __host__ void operator/=(float4 &v, float a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
  v.w /= a;
}

inline __device__ __host__ float2 operator*(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
// divide
inline __device__ __host__ float2 operator/(float2 a, float s) { return make_float2(a.x / s, a.y / s); }

inline __device__ __host__ float2 operator/(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }

inline __device__ __host__ int2 operator/(int2 a, int2 b) { return make_int2(a.x / b.x, a.y / b.y); }

inline __device__ __host__ bool operator==(float2 a, float2 b) { return (a.x == b.x && a.y == b.y); }

inline __host__ void operator*=(double3 &v, double a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
}

inline __host__ void operator/=(double3 &v, double a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
}

inline __device__ __host__ float length(float2 v) {
#ifdef __CUDA_ARCH__
  return __fsqrt_rn(v.x * v.x + v.y * v.y);
#else
  return sqrtf(v.x * v.x + v.y * v.y);
#endif
}

inline __device__ __host__ float length(float3 v) {
#ifdef __CUDA_ARCH__
  return __fsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);
#else
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
#endif
}

inline __device__ __host__ float invLength(float2 v) {
#ifdef __CUDA_ARCH__
  return rsqrtf(v.x * v.x + v.y * v.y);
#else
  return 1.0f / sqrtf(v.x * v.x + v.y * v.y);
#endif
}

inline __device__ __host__ float invLength(float3 v) {
#ifdef __CUDA_ARCH__
  return rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
#else
  return 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
#endif
}

inline __device__ __host__ float clampf(const float input, const float minValue, const float maxValue) {
#ifdef __CUDA_ARCH__
  return fmin(fmax(input, minValue), maxValue);
#else
  return input < minValue ? minValue : (input > maxValue ? maxValue : input);
#endif
}

inline __device__ __host__ float2 normalize(float2 v) {
  float norm = invLength(v);
  v.x *= norm;
  v.y *= norm;
  return v;
}

inline __device__ __host__ float3 normalize(float3 v) {
  float norm = invLength(v);
  v.x *= norm;
  v.y *= norm;
  v.z *= norm;
  return v;
}

inline __device__ __host__ float2 clampf(const float2 input, const float minValue, const float maxValue) {
  float2 output;
  output.x = clampf(input.x, minValue, maxValue);
  output.y = clampf(input.y, minValue, maxValue);
  return output;
}
