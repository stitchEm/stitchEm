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

namespace VideoStitch {

inline __device__ __host__ bool inRange(const int2 coord, const int2 size) {
  return (coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y);
}

inline __device__ __host__ bool inRange(const float2 coord, const int2 size) {
  return (coord.x >= 0 && ceilf(coord.x) < size.x && coord.y >= 0 && ceilf(coord.y) < size.y);
}

namespace Image {

inline __device__ __host__ float3 rgbToXYZ(const float3 rgb) {
  // https://en.wikipedia.org/wiki/SRGB
  const float r = rgb.x;
  const float g = rgb.y;
  const float b = rgb.z;
  return make_float3(0.4124f * r + 0.3576f * g + 0.1805f * b, 0.2126f * r + 0.7152f * g + 0.0722f * b,
                     0.0193f * r + 0.1192f * g + 0.9505f * b);
}

inline __device__ __host__ float3 xyzToRGB(const float3 xyz) {
  const float x = xyz.x;
  const float y = xyz.y;
  const float z = xyz.z;
  return make_float3(3.2404f * x - 1.5371f * y - 0.4985f * z, -0.9692f * x + 1.8760f * y + 0.0415f * z,
                     0.0556f * x - 0.2040f * y + 1.0572f * z);
}

// Lab
inline __device__ __host__ float3 xyzToLab(const float3 xyz) {
  // https://en.wikipedia.org/wiki/Lab_color_space
  float var_X = xyz.x / (95.047f / 100.0f);
  float var_Y = xyz.y / (100.000f / 100.0f);
  float var_Z = xyz.z / (108.883f / 100.0f);
  if (var_X > 0.008856f)
    var_X = powf(var_X, (1.0f / 3.0f));
  else
    var_X = (7.787f * var_X) + (16.0f / 116.0f);
  if (var_Y > 0.008856f)
    var_Y = powf(var_Y, (1.0f / 3.0f));
  else
    var_Y = (7.787f * var_Y) + (16.0f / 116.0f);
  if (var_Z > 0.008856f)
    var_Z = powf(var_Z, (1.0f / 3.0f));
  else
    var_Z = (7.787f * var_Z) + (16.0f / 116.0f);
  const float l = (116.0f * var_Y) - 16.0f;
  const float a = 500.0f * (var_X - var_Y);
  const float b = 200.0f * (var_Y - var_Z);

  return make_float3(l, a, b);
}

inline __device__ __host__ float3 rgbToLab(const float3 rgb) { return xyzToLab(rgbToXYZ(rgb)); }

inline __device__ __host__ float rgbToLuminance(const float3 rgb) {
  return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

inline __device__ __host__ float3 labToXYZ(const float3 lab) {
  float var_Y = (lab.x + 16.0f) / 116.0f;
  float var_X = lab.y / 500.0f + var_Y;
  float var_Z = var_Y - lab.z / 200.0f;

  if (powf(var_Y, 3.0f) > 0.008856f)
    var_Y = powf(var_Y, 3.0f);
  else
    var_Y = (var_Y - 16.0f / 116.0f) / 7.787f;
  if (powf(var_X, 3.0f) > 0.008856f)
    var_X = powf(var_X, 3.0f);
  else
    var_X = (var_X - 16.0f / 116.0f) / 7.787f;
  if (powf(var_Z, 3.0f) > 0.008856f)
    var_Z = powf(var_Z, 3.0f);
  else
    var_Z = (var_Z - 16.0f / 116.0f) / 7.787f;

  const float x = var_X * (95.047f / 100.0f);
  const float y = var_Y * (100.000f / 100.0f);
  const float z = var_Z * (108.883f / 100.0f);

  return make_float3(x, y, z);
}

inline __device__ __host__ float3 labToRGB(const float3 lab) { return xyzToRGB(labToXYZ(lab)); }

inline __device__ __host__ uint32_t sqrDist(int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

inline __device__ __host__ int32_t clamp8(int32_t c) {
#ifdef __CUDACC__
  return min(255, max(0, c));
#else
  return c < 0 ? 0 : (c > 255 ? 255 : c);
#endif
}

/**
 * RGBA format: 8 bits per component, AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR
 */
class RGBABase {
 public:
  typedef uint32_t T;

  /**
   * Get the red component for a pixel.
   * @param v RGBA8888 packed pixel.
   */
  static inline __device__ __host__ uint32_t r(uint32_t v) { return v & (uint32_t)0xff; }

  /**
   * Get the green component for a pixel.
   * @param v RGBA8888 packed pixel.
   */
  static inline __device__ __host__ uint32_t g(uint32_t v) { return (v >> 8) & (uint32_t)0xff; }

  /**
   * Get the blue component for a pixel.
   * @param v RGBA8888 packed pixel.
   */
  static inline __device__ __host__ uint32_t b(uint32_t v) { return (v >> 16) & (uint32_t)0xff; }

 private:
  RGBABase();
};

/**
 * RGBA format, with alpha.
 */
class RGBA : public RGBABase {
 public:
  /**
   * Get the alpha component for a pixel.
   * @param v RGBA8888 packed pixel.
   */
  static inline __device__ __host__ uint32_t a(uint32_t v) { return (v >> 24) & (uint32_t)0xff; }

  /**
   * Pack RGBA values into a 32 bits pixel as .
   * @param r Red component. Between 0 and 255.
   * @param g Green component. Between 0 and 255.
   * @param b Blue component. Between 0 and 255.
   * @param a Alpha component. Between 0 and 255.
   */
  static inline __device__ __host__ uint32_t pack(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    return (a << 24) | (b << 16) | (g << 8) | r;
  }

 private:
  RGBA();
};

/**
 * RGBA solid: Alpha is always 0xff.
 */
class RGBASolid : public RGBABase {
 public:
  /**
   * Get the alpha component for a pixel.
   * @param v RGBA8888 packed pixel.
   */
  static inline __device__ __host__ uint32_t a(uint32_t /*v*/) { return (uint32_t)0xff; }

  /**
   * Pack RGBA values into a 32 bits pixel as .
   * @param r Red component. Between 0 and 255.
   * @param g Green component. Between 0 and 255.
   * @param b Blue component. Between 0 and 255.
   * @param a Alpha component. Between 0 and 255.
   */
  static inline __device__ __host__ uint32_t pack(uint32_t r, uint32_t g, uint32_t b, uint32_t /*a*/) {
    return 0xff000000u | (b << 16) | (g << 8) | r;
  }

 private:
  RGBASolid();
};

/**
 * RGBA64 format: 16 bits per component, AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR
 * This is a class for templating.
 */
class RGBA64 {
 public:
  typedef uint64_t T;

  /**
   * Get the red component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  static inline __device__ __host__ uint32_t r(uint64_t v) { return (uint32_t)v & (uint32_t)0xffff; }

  /**
   * Get the green component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  static inline __device__ __host__ uint32_t g(uint64_t v) { return (uint32_t)(v >> 16) & (uint32_t)0xffff; }

  /**
   * Get the blue component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  static inline __device__ __host__ uint32_t b(uint64_t v) { return (uint32_t)(v >> 32) & (uint32_t)0xffff; }

  /**
   * Get the alpha component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  static inline __device__ __host__ uint32_t a(uint64_t v) { return (uint32_t)(v >> 48) & (uint32_t)0xffff; }

  /**
   * Pack RGBA values into a 32 bits pixel as .
   * @param r Red component. Between 0 and (1<<16 - 1).
   * @param g Green component. Between 0 and (1<<16 - 1).
   * @param b Blue component. Between 0 and (1<<16 - 1).
   * @param a Alpha component. Between 0 and (1<<16 - 1).
   */
  static inline __device__ __host__ uint64_t pack(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    return ((uint64_t)a << 48) | ((uint64_t)b << 32) | ((uint64_t)g << 16) | (uint64_t)r;
  }

 private:
  RGBA64();
};

/**
 * 8-bit monochrome Y component.
 */
class MonoY8 {
 public:
  /**
   * Get the red component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  //   static inline __device__ __host__ uint32_t r(unsigned char v) {
  //     return v;
  //   }

  /**
   * Get the green component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  //   static inline __device__ __host__ uint32_t g(unsigned char v) {
  //     return v;
  //   }

  /**
   * Get the blue component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  //   static inline __device__ __host__ uint32_t b(unsigned char v) {
  //     return v;
  //   }

  /**
   * Get the alpha component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  //   static inline __device__ __host__ uint32_t a(uint64_t /*v*/) {
  //     return 255;
  //   }

  /**
   * Pack RGBA values into a 32 bits pixel as .
   * @param r Red component. Between 0 and (255).
   * @param g Green component. Between 0 and (255).
   * @param b Blue component. Between 0 and (255).
   * @param a Alpha component. Ignored.
   */
  static inline __device__ __host__ unsigned char pack(uint32_t r, uint32_t g, uint32_t b, uint32_t /*a*/) {
    return (unsigned char)(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
  }

 private:
  MonoY8();
};

/**
 * Solid RGB.
 */
class RGBSolid {
 public:
  /**
   * Get the red component for a pixel.
   * @param v pixel value.
   */
  static inline __device__ __host__ uint32_t r(uchar3 v) { return v.x; }

  /**
   * Get the green component for a pixel.
   * @param v pixel value.
   */
  static inline __device__ __host__ uint32_t g(uchar3 v) { return v.y; }

  /**
   * Get the blue component for a pixel.
   * @param v pixel value.
   */
  static inline __device__ __host__ uint32_t b(uchar3 v) { return v.z; }

  /**
   * Get the alpha component for a pixel.
   * @param v RGBA64 packed pixel.
   */
  static inline __device__ __host__ uint32_t a(char3 /*v*/) { return 255; }

  /**
   * Pack RGBA values into a 32 bits pixel as .
   * @param r Red component. Between 0 and 255.
   * @param g Green component. Between 0 and 255.
   * @param b Blue component. Between 0 and 255.
   * @param a Alpha component. Ignored.
   */
  static inline __device__ __host__ uchar3 pack(uint32_t r, uint32_t g, uint32_t b, uint32_t /*a*/) {
    uchar3 v;
    v.x = (unsigned char)r;
    v.y = (unsigned char)g;
    v.z = (unsigned char)b;
    return v;
  }

 private:
  RGBSolid();
};

/**
 * RGB210 format: 1 alpha bit, 3 * signed 9-bit colors: A_RRRRRRRRRRGGGGGGGGGGBBBBBBBBBB
 * This is a class for templating.
 */
class RGB210 {
 public:
  typedef uint32_t T;

  /**
   * Get the red component for a packed pixel.
   * @param v RGBA210 packed pixel.
   */
  static inline __device__ __host__ int32_t r(uint32_t v) {
    int32_t const m = 0x200;  // 1 << (10 - 1)
    int32_t x = v & (uint32_t)0x3ff;
    return (x ^ m) - m;
  }

  /**
   * Get the green component for a packed pixel.
   * @param v RGBA210 packed pixel.
   */
  static inline __device__ __host__ int32_t g(uint32_t v) {
    int32_t const m = 0x200;  // 1 << (10 - 1)
    int32_t x = (v >> 10) & (uint32_t)0x3ff;
    return (x ^ m) - m;  // absolute value ???
  }

  /**
   * Get the blue component for a packed pixel.
   * @param v RGBA210 packed pixel.
   */
  static inline __device__ __host__ int32_t b(uint32_t v) {
    int32_t const m = 0x200;  // 1 << (10 - 1)
    int32_t x = (v >> 20) & (uint32_t)0x3ff;
    return (x ^ m) - m;  // absolute value ???
  }

  /**
   * Get the alpha component for a packed pixel.
   * @param v RGBA210 packed pixel.
   * @note This is guaranteed to return only 0 or 1.
   */
  static inline __device__ __host__ int32_t a(uint32_t v) { return (int32_t)(v >> 31) == 0 ? 0 : 255; }

  /**
   * Pack RGBA values into a 32 bits pixel.
   * @param r Red component. Between -511 and 511.
   * @param g Green component. Between -511 and 511.
   * @param b Blue component. Between -511 and 511.
   * @param a Alpha component. If <= 0, transparent. Else, solid.
   */
  static inline __device__ __host__ uint32_t pack(int32_t r, int32_t g, int32_t b, int32_t a) {
    return (((uint32_t) !!(a > 0)) * (uint32_t)0x80000000) | (((uint32_t)b & 0x3ff) << 20) |
           (((uint32_t)g & 0x3ff) << 10) | ((uint32_t)r & 0x3ff);
  }

 private:
  RGB210();  // Static only;
};

}  // namespace Image
}  // namespace VideoStitch
