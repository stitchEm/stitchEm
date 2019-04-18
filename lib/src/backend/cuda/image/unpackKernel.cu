// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/imageOps.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Image {

inline __device__ uint32_t YRGBDiffToRGBA(unsigned char y, const int3& rgbDiff) {
  const int32_t ya = (1164 * (y - 16)) / 1000;
  return RGBA::pack(clamp8(ya + rgbDiff.x), clamp8(ya + rgbDiff.y), clamp8(ya + rgbDiff.z), 0xff);
}

#define nv12_surface_write surface_write_i

#include "../gpuKernelDef.h"
#include "backend/common/image/unpack.gpu"

// ---------------------------- Output -----------------------------

__global__ void unpackKernelGrayscale(unsigned char* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width,
                                      unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uint32_t val;
    surf2Dread(&val, src, x * sizeof(uint32_t), y);
    int32_t r = RGBA::r(val);
    int32_t g = RGBA::g(val);
    int32_t b = RGBA::b(val);
    dst[y * pitch + x] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
  }
}

__global__ void unpackSourceKernelRGBA(uint32_t* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width,
                                       unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    // yeah, we could use a memcpy
    uint32_t val;
    surf2Dread(&val, src, x * sizeof(uint32_t), y);
    dst[y * pitch + x] = val;
  }
}

__global__ void unpackKernelRGB(unsigned char* __restrict__ dst, unsigned pitch, const uint32_t* __restrict__ src,
                                unsigned width, unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    const uint32_t val = src[y * width + x];
    dst[y * pitch + 3 * x] = RGBA::r(val);
    dst[y * pitch + 3 * x + 1] = RGBA::g(val);
    dst[y * pitch + 3 * x + 2] = RGBA::b(val);
  }
}

__global__ void unpackSourceKernelRGB(unsigned char* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width,
                                      unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uint32_t val;
    surf2Dread(&val, src, x * sizeof(uint32_t), y);
    dst[y * pitch + 3 * x] = RGBA::r(val);
    dst[y * pitch + 3 * x + 1] = RGBA::g(val);
    dst[y * pitch + 3 * x + 2] = RGBA::b(val);
  }
}

__global__ void unpackSourceKernelF32C1(float* dst, unsigned pitch, const cudaSurfaceObject_t src, unsigned width,
                                        unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    // yeah, we could use a memcpy
    float val;
    surf2Dread(&val, src, x * sizeof(float), y);
    dst[y * pitch + x] = val;
  }
}

__global__ void unpackSourceKernelGrayscale16(uint16_t* dst, unsigned pitch, const cudaSurfaceObject_t src,
                                              unsigned width, unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    float val;
    surf2Dread(&val, src, x * sizeof(float), y);
    const float inMilliMeters = val * 1000.f;
    const uint16_t u16 = (uint16_t)max(0.f, min((float)USHRT_MAX, round(inMilliMeters)));
    dst[y * pitch + x] = u16;
  }
}

__global__ void unpackKernelDepth(unsigned char* __restrict__ yDst, unsigned yPitch, unsigned char* __restrict__ uDst,
                                  unsigned uPitch, unsigned char* __restrict__ vDst, unsigned vPitch,
                                  const float* __restrict__ src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  if (x < width && y < height) {
    int32_t u = 0;
    int32_t v = 0;

#pragma unroll
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        const float depth = src[(y + j) * width + x + i];
        // convert to millimeters and truncate
        unsigned int val = min(__float2uint_rn(depth * 1000.f), 65279);
        // encode
        yDst[(y + j) * yPitch + x + i] = (unsigned char)(val / 256);
        int cu = val % 512;
        int cv = (val + 384) % 512;
        if (cu >= 256) {
          u += (unsigned char)(511 - cu);
        } else {
          u += (unsigned char)cu;
        }
        if (cv >= 256) {
          v += (unsigned char)(511 - cv);
        } else {
          v += (unsigned char)cv;
        }
      }
    }
    uDst[(y * uPitch + x) / 2] = (u + 2) / 4;
    vDst[(y * vPitch + x) / 2] = (v + 2) / 4;
  }
}

__global__ void unpackSourceKernelDepth(unsigned char* __restrict__ yDst, unsigned yPitch,
                                        unsigned char* __restrict__ uDst, unsigned uPitch,
                                        unsigned char* __restrict__ vDst, unsigned vPitch,
                                        const cudaSurfaceObject_t src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  if (x < width && y < height) {
    int32_t u = 0;
    int32_t v = 0;

#pragma unroll
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        float depth;
        surf2Dread(&depth, src, (x + i) * sizeof(float), y + j);
        // convert to millimeters and truncate
        unsigned int val = min(__float2uint_rn(depth * 1000.f), 65279);
        // encode
        yDst[(y + j) * yPitch + x + i] = (unsigned char)(val / 256);
        int cu = val % 512;
        int cv = (val + 384) % 512;
        if (cu >= 256) {
          u += (unsigned char)(511 - cu);
        } else {
          u += (unsigned char)cu;
        }
        if (cv >= 256) {
          v += (unsigned char)(511 - cv);
        } else {
          v += (unsigned char)cv;
        }
      }
    }
    uDst[(y * uPitch + x) / 2] = (u + 2) / 4;
    vDst[(y * vPitch + x) / 2] = (v + 2) / 4;
  }
}

/**
 * This kernel converts the buffer from RGBA to planar 12 bits 4:2:0 (YV12) out-of-place.
 * The conversion is undefined for pixels with 0 alpha.
 *
 * Y0 Y1 Y2 Y3
 * ...
 * U0 U1
 * ...
 * V0 V1
 * ...
 */
__global__ void unpackKernelYV12(unsigned char* __restrict__ yDst, unsigned yPitch, unsigned char* __restrict__ uDst,
                                 unsigned uPitch, unsigned char* __restrict__ vDst, unsigned vPitch,
                                 const uint32_t* __restrict__ src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned sy = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
  if (sx < width && sy < height) {
    int32_t u = 0;
    int32_t v = 0;
    {
      uint32_t val = src[sy * width + sx];
      int32_t r = RGBA::r(val);
      int32_t g = RGBA::g(val);
      int32_t b = RGBA::b(val);
      int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        uint32_t val = src[sy * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val = src[(sy + 1) * width + sx];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val = src[(sy + 1) * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uDst[(sy * uPitch + sx) / 2] = u / 4;
      vDst[(sy * vPitch + sx) / 2] = v / 4;
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        uint32_t val = src[sy * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = u / 2;
        vDst[(sy * vPitch + sx) / 2] = v / 2;
      }
      __syncthreads();
      if (sy + 1 < height) {
        uint32_t val = src[(sy + 1) * width + sx];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = u / 2;
        vDst[(sy * vPitch + sx) / 2] = v / 2;
      }
    }
  }
}

__global__ void unpackSourceKernelYV12(unsigned char* __restrict__ yDst, unsigned yPitch,
                                       unsigned char* __restrict__ uDst, unsigned uPitch,
                                       unsigned char* __restrict__ vDst, unsigned vPitch, const cudaSurfaceObject_t src,
                                       unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned sy = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
  if (sx < width && sy < height) {
    int32_t u = 0;
    int32_t v = 0;
    {
      uint32_t val;
      surf2Dread(&val, src, sx * sizeof(uint32_t), sy);
      int32_t r = clamp8(RGBA::r(val));
      int32_t g = clamp8(RGBA::g(val));
      int32_t b = clamp8(RGBA::b(val));
      int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy);
        int32_t r = clamp8(RGBA::r(val));
        int32_t g = clamp8(RGBA::g(val));
        int32_t b = clamp8(RGBA::b(val));
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val;
        surf2Dread(&val, src, sx * sizeof(uint32_t), sy + 1);
        int32_t r = clamp8(RGBA::r(val));
        int32_t g = clamp8(RGBA::g(val));
        int32_t b = clamp8(RGBA::b(val));
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy + 1);
        int32_t r = clamp8(RGBA::r(val));
        int32_t g = clamp8(RGBA::g(val));
        int32_t b = clamp8(RGBA::b(val));
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uDst[(sy * uPitch + sx) / 2] = u / 4;
      vDst[(sy * vPitch + sx) / 2] = v / 4;
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy);
        int32_t r = clamp8(RGBA::r(val));
        int32_t g = clamp8(RGBA::g(val));
        int32_t b = clamp8(RGBA::b(val));
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = u / 2;
        vDst[(sy * vPitch + sx) / 2] = v / 2;
      }
      __syncthreads();
      if (sy + 1 < height) {
        uint32_t val;
        surf2Dread(&val, src, sx * sizeof(uint32_t), sy + 1);
        int32_t r = clamp8(RGBA::r(val));
        int32_t g = clamp8(RGBA::g(val));
        int32_t b = clamp8(RGBA::b(val));
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = u / 2;
        vDst[(sy * vPitch + sx) / 2] = v / 2;
      }
    }
  }
}

/**
 * This kernel converts the buffer from RGBA to interleaved YUV420 (NV12) out-of-place.
 * The conversion is undefined for pixels with 0 alpha.
 *
 * Y0 Y1 Y2 Y3
 * ...
 * U0 V0 U1 V1
 * ...
 */
__global__ void unpackKernelNV12(unsigned char* __restrict__ yDst, unsigned yPitch, unsigned char* __restrict__ uvDst,
                                 unsigned uvPitch, const uint32_t* __restrict__ src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned sy = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  if (sx < width && sy < height) {
    int32_t u = 0;
    int32_t v = 0;
    {
      uint32_t val = src[sy * width + sx];
      int32_t r = RGBA::r(val);
      int32_t g = RGBA::g(val);
      int32_t b = RGBA::b(val);
      int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }

    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        uint32_t val = src[sy * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val = src[(sy + 1) * width + sx];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val = src[(sy + 1) * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uvDst[(sy * uvPitch) / 2 + sx] = u / 4;
      uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 4;
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        uint32_t val = src[sy * width + sx + 1];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * uvPitch) / 2 + sx] = u / 2;
        uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 2;
      } else if (sy + 1 < height) {
        uint32_t val = src[(sy + 1) * width + sx];
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * uvPitch) / 2 + sx] = u / 2;
        uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 2;
      }
    }
  }
}

/**
 * This kernel converts the buffer from RGBA to interleaved YUV420 (NV12) out-of-place.
 * The conversion is undefined for pixels with 0 alpha.
 *
 * Y0 Y1 Y2 Y3
 * ...
 * U0 V0 U1 V1
 * ...
 */
__global__ void unpackSourceKernelNV12(unsigned char* __restrict__ yDst, unsigned yPitch,
                                       unsigned char* __restrict__ uvDst, unsigned uvPitch,
                                       const cudaSurfaceObject_t src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned sy = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  if (sx < width && sy < height) {
    int32_t u = 0;
    int32_t v = 0;
    {
      uint32_t val;
      surf2Dread(&val, src, sx * sizeof(uint32_t), sy);
      int32_t r = RGBA::r(val);
      int32_t g = RGBA::g(val);
      int32_t b = RGBA::b(val);
      int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }

    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy);
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val;
        surf2Dread(&val, src, sx * sizeof(uint32_t), sy + 1);
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy + 1);
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uvDst[(sy * uvPitch) / 2 + sx] = u / 4;
      uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 4;
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        uint32_t val;
        surf2Dread(&val, src, (sx + 1) * sizeof(uint32_t), sy);
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * uvPitch) / 2 + sx] = u / 2;
        uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 2;
      } else if (sy + 1 < height) {
        uint32_t val;
        surf2Dread(&val, src, sx * sizeof(uint32_t), sy + 1);
        int32_t r = RGBA::r(val);
        int32_t g = RGBA::g(val);
        int32_t b = RGBA::b(val);
        int32_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * uvPitch) / 2 + sx] = u / 2;
        uvDst[(sy * uvPitch) / 2 + sx + 1] = v / 2;
      }
    }
  }
}

/**
 * This kernel converts the buffer from RGBA to YUY2 out-of-place.
 * Pixels are all given full solidness (max alpha).
 */
__global__ void unpackYUY2Kernel(unsigned char* __restrict__ dst, unsigned pitch, const uint32_t* __restrict__ src,
                                 unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width / 2 && y < height) {
    uint32_t val0 = src[y * width + 2 * x];
    int32_t r0 = RGBA::r(val0);
    int32_t g0 = RGBA::g(val0);
    int32_t b0 = RGBA::b(val0);
    uint32_t val1 = src[y * width + 2 * x + 1];
    int32_t r1 = RGBA::r(val1);
    int32_t g1 = RGBA::g(val1);
    int32_t b1 = RGBA::b(val1);

    unsigned char y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
    unsigned char y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    unsigned char u = ((-38 * r0 - 74 * g0 + 112 * b0 + 128) >> 8) + 128;
    unsigned char v = ((112 * r0 - 94 * g0 - 18 * b0 + 128) >> 8) + 128;

    dst[y * pitch + 4 * x] = y0;
    dst[y * pitch + 4 * x + 1] = u;
    dst[y * pitch + 4 * x + 2] = y1;
    dst[y * pitch + 4 * x + 3] = v;
  }
}

/**
 * This kernel converts the buffer from RGBA to UYVY out-of-place.
 * Pixels are all given full solidness (max alpha).
 */
__global__ void unpackUYVYKernel(unsigned char* __restrict__ dst, unsigned pitch, const uint32_t* __restrict__ src,
                                 unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width / 2 && y < height) {
    uint32_t val0 = src[y * width + 2 * x];
    int32_t r0 = RGBA::r(val0);
    int32_t g0 = RGBA::g(val0);
    int32_t b0 = RGBA::b(val0);
    uint32_t val1 = src[y * width + 2 * x + 1];
    int32_t r1 = RGBA::r(val1);
    int32_t g1 = RGBA::g(val1);
    int32_t b1 = RGBA::b(val1);

    unsigned char y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
    unsigned char y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    unsigned char u = ((-38 * r0 - 74 * g0 + 112 * b0 + 128) >> 8) + 128;
    unsigned char v = ((112 * r0 - 94 * g0 - 18 * b0 + 128) >> 8) + 128;

    dst[y * pitch + 4 * x] = u;
    dst[y * pitch + 4 * x + 1] = y0;
    dst[y * pitch + 4 * x + 2] = v;
    dst[y * pitch + 4 * x + 3] = y1;
  }
}

/**
 * This kernel converts the buffer from RGBA to 10 bits planar YUV422 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * 10 bits values are padded to 16 bits.
 */
__global__ void unpackYUV422P10Kernel(uint16_t* __restrict__ yDst, unsigned yPitch, uint16_t* __restrict__ uDst,
                                      unsigned uPitch, uint16_t* __restrict__ vDst, unsigned vPitch,
                                      const uint32_t* __restrict__ src, unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width / 2 && y < height) {
    uint32_t val0 = src[y * width + 2 * x];
    int32_t r0 = RGBA::r(val0);
    int32_t g0 = RGBA::g(val0);
    int32_t b0 = RGBA::b(val0);
    uint32_t val1 = src[y * width + 2 * x + 1];
    int32_t r1 = RGBA::r(val1);
    int32_t g1 = RGBA::g(val1);
    int32_t b1 = RGBA::b(val1);
    uint32_t u = 0, v = 0;
    int32_t y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16 << 2;
    int32_t y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16 << 2;
    u += ((-38 * r0 - 74 * g0 + 112 * b0 + 128) >> 8) + 128 << 2;
    u += ((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128 << 2;
    v += ((112 * r0 - 94 * g0 - 18 * b0 + 128) >> 8) + 128 << 2;
    v += ((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128 << 2;

    yDst[y * yPitch + 2 * x] = y0;
    yDst[y * yPitch + 2 * x + 1] = y1;
    uDst[y * uPitch + x] = u / 2;
    vDst[y * vPitch + x] = v / 2;
  }
}

__global__ void unpackMonoKernelYUV420P(unsigned char* __restrict__ dst, const unsigned char* __restrict__ src,
                                        unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned sy = blockIdx.y * blockDim.y + threadIdx.y;

  if (sx < width / 2 && sy < height / 2) {
    {
      const unsigned i = (2 * sy) * width + 2 * sx;
      dst[i] = src[i];
    }
    {
      const unsigned i = (2 * sy) * width + 2 * sx + 1;
      dst[i] = src[i];
    }
    {
      const unsigned i = (2 * sy + 1) * width + 2 * sx;
      dst[i] = src[i];
    }
    {
      const unsigned i = (2 * sy + 1) * width + 2 * sx + 1;
      dst[i] = src[i];
    }
  }
}

// ---------------------------- Input -----------------------------

/**
 * This kernel converts the buffer from BGRU8888 (where 'U' stands for 'unused') to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha)
 */
__global__ void convertBGRUToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                        unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    dst[i] = RGBA::pack(src[4 * i + 2], src[4 * i + 1], src[4 * i], 0xff);
  }
}

/**
 * This kernel converts the buffer from RGB to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha)
 */
__global__ void convertRGBToRGBAKernel(cudaSurfaceObject_t dst, const unsigned char* src, unsigned width,
                                       unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    surf2Dwrite(RGBA::pack(src[3 * i], src[3 * i + 1], src[3 * i + 2], 0xff), dst, x * sizeof(uint32_t), y);
  }
}

__global__ void convertRGB210ToRGBAKernel(cudaSurfaceObject_t dst, const uint32_t* src, unsigned width,
                                          unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    uint32_t v = src[y * width + x];
    surf2Dwrite(RGBA::pack(clamp8(RGB210::r(v)), clamp8(RGB210::g(v)), clamp8(RGB210::b(v)), RGB210::a(v)), dst,
                x * sizeof(uint32_t), y);
  }
}

__device__ unsigned loadBayerPattern(const unsigned char* __restrict__ src, unsigned width, unsigned height,
                                     unsigned char* sharedSrc, unsigned srcX, unsigned srcY) {
  // The shared memory uses the same pattern as src.
  // There are (2 * blockDim.x + 2) * (2 * blockDim.y + 2) bytes to load
  // (we need an extra layer outside the current zone for interpolation).
  const unsigned sharedBlockWidth = blockDim.x + 1;  // +1: one half block left and one half-block right.
  const unsigned sharedWidth = 2 * sharedBlockWidth;

  // Start with interior blocks.
  if (srcX < width && srcY < height) {
    // The access pattern is the same as during interpolation, meaning that each thread issues two (coalesced) reads: RG
    // then GB.
    // TODO: try out a different loading pattern: each thread loads 4 consecutive bytes in memory.
    //       This would reduce the number of coalesced reads to 1 instead of 2.
    //       Note that this would not be the same pattern as during interpolation.
    const int srcBase = width * srcY + srcX;
    const int sharedBase = sharedWidth + 1 + 2 * (sharedWidth * threadIdx.y + threadIdx.x);
    // The compiler should be able to optimize that in only 2 coalesced single word reads.
    // If it can't, accesses are still coalesced, but there are 4 accesses instead of one.
    sharedSrc[sharedBase] = src[srcBase];
    sharedSrc[sharedBase + 1] = src[srcBase + 1];
    sharedSrc[sharedBase + sharedWidth] = src[srcBase + width];
    sharedSrc[sharedBase + sharedWidth + 1] = src[srcBase + width + 1];
  }
  // Now load the boundary
  if (threadIdx.y == 0 && srcX < width) {
    // Top
    {
      const int sharedBase = 1 + 2 * threadIdx.x;
      const int srcBase = srcY > 0 ?
                                   // Normal case.
                              width * (srcY - 1) + srcX
                                   :
                                   // The previous row is outside the image, constant boundary condition.
                              width + srcX;
      sharedSrc[sharedBase] = src[srcBase];
      sharedSrc[sharedBase + 1] = src[srcBase + 1];
    }
    // Bottom
    {
      int srcBoundaryRow;
      int sharedBoundaryRow;
      if (srcY + 2 * blockDim.y < height) {
        // Normal case, extra row is within image.
        srcBoundaryRow = srcY + 2 * blockDim.y;
        sharedBoundaryRow = 2 * blockDim.y;
      } else {
        // The next row is outside the image, constant boundary condition.
        srcBoundaryRow = height - 2;
        sharedBoundaryRow = height - srcY;
      }
      const int srcBase = width * srcBoundaryRow + srcX;
      const int sharedBase = sharedWidth + 1 + sharedWidth * sharedBoundaryRow + 2 * threadIdx.x;
      sharedSrc[sharedBase] = src[srcBase];
      sharedSrc[sharedBase + 1] = src[srcBase + 1];
    }
  }
  if (threadIdx.x == 0 && srcY < height) {
    // Left
    {
      const int sharedBase = sharedWidth + 2 * sharedWidth * threadIdx.y;
      const int srcBase = srcX > 0 ?
                                   // Normal case.
                              width * srcY + srcX - 1
                                   :
                                   // The previous col is outside the image, constant boundary condition.
                              width * srcY + 1;
      sharedSrc[sharedBase] = src[srcBase];
      sharedSrc[sharedBase + sharedWidth] = src[srcBase + width];
    }
    // Right
    {
      int srcBoundaryCol;
      int sharedBoundaryCol;
      if (srcX + 2 * blockDim.x < width) {
        // Normal case, extra col is within image.
        srcBoundaryCol = srcX + 2 * blockDim.x;
        sharedBoundaryCol = 2 * blockDim.x;
      } else {
        // The next col is outside the image, constant boundary condition.
        srcBoundaryCol = width - 2;
        sharedBoundaryCol = width - srcX;
      }
      const int srcBase = width * srcY + srcBoundaryCol;
      const int sharedBase = sharedWidth + 1 + 2 * sharedWidth * threadIdx.y + sharedBoundaryCol;
      sharedSrc[sharedBase] = src[srcBase];
      sharedSrc[sharedBase + sharedWidth] = src[srcBase + width];
    }
  }
  // And the corners.
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // Due to the assymetry, only the top left and bottom right corner are ever used (see the test for an example).
    // Top left
    {
      const int srcBoundaryCol = srcX > 0 ? srcX - 1 : 1;
      const int srcBoundaryRow = srcY > 0 ? srcY - 1 : 1;
      const int srcBase = width * srcBoundaryRow + srcBoundaryCol;
      sharedSrc[0] = src[srcBase];
    }
    // Bottom right.
    {
      int srcBoundaryCol;
      int sharedBoundaryCol;
      if (srcX + 2 * blockDim.x < width) {
        // Normal case, extra col is within image.
        srcBoundaryCol = srcX + 2 * blockDim.x;
        sharedBoundaryCol = 2 * blockDim.x;
      } else {
        // The next col is outside the image, constant boundary condition.
        srcBoundaryCol = width - 2;
        sharedBoundaryCol = width - srcX;
      }
      int srcBoundaryRow;
      int sharedBoundaryRow;
      if (srcY + 2 * blockDim.y < height) {
        // Normal case, extra row is within image.
        srcBoundaryRow = srcY + 2 * blockDim.y;
        sharedBoundaryRow = 2 * blockDim.y;
      } else {
        // The next row is outside the image, constant boundary condition.
        srcBoundaryRow = height - 2;
        sharedBoundaryRow = height - srcY;
      }
      const int srcBase = width * srcBoundaryRow + srcBoundaryCol;
      const int sharedBase = sharedWidth + 1 + sharedWidth * sharedBoundaryRow + sharedBoundaryCol;
      sharedSrc[sharedBase] = src[srcBase];
    }
  }

  return sharedWidth;
}

/**
 * This kernel converts the buffer from Bayer-filtered RGGB to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * Uses bilinear interpolation within color planes.
 *
 * Each thread handles a 2*2 RGGB pixel block. The interpolation support is the 4*4 pixel block centered around the 2*2
 * block. Globally each thread block needs an extra pixel around itself.
 *
 */
__global__ void convertBayerRGGBToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                             unsigned width, unsigned height) {
  // x and y are the 2*2 block ids.
  const unsigned srcX = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned srcY = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  // Load the data to shared memory.
  extern __shared__ unsigned char sharedSrc[];
  const unsigned sharedWidth = loadBayerPattern(src, width, height, sharedSrc, srcX, srcY);
  __syncthreads();

  if (srcX < width && srcY < height) {
    const int sharedBase = sharedWidth + 1 + 2 * (sharedWidth * threadIdx.y + threadIdx.x);
    // Top-left component;
    dst[srcY * width + srcX] = RGBA::pack(
        sharedSrc[sharedBase],  // Red is given
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1] +
         (int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + sharedWidth]) /
            4,  // Green is 4-tap straight (+)
        ((int32_t)sharedSrc[sharedBase - 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 - sharedWidth] +
         (int32_t)sharedSrc[sharedBase - 1 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            4,  // Blue is 4-tap 45° rotated (x)
        255);
    // Top-right component;
    dst[srcY * width + srcX + 1] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2]) / 2,  // Red is 2-tap horizontal
        sharedSrc[sharedBase + 1],                                                  // Green is given
        ((int32_t)sharedSrc[sharedBase + 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            2,  // Blue is 2-tap vertical
        255);
    // Bottom-left component;
    dst[(srcY + 1) * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2 * sharedWidth]) /
            2,                                // Red is 2-tap vertical
        sharedSrc[sharedBase + sharedWidth],  // Green is given
        ((int32_t)sharedSrc[sharedBase + sharedWidth - 1] + (int32_t)sharedSrc[sharedBase + sharedWidth + 1]) /
            2,  // Blue is 2-tap horizontal
        255);
    // Bottom-right component
    dst[(srcY + 1) * width + srcX + 1] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2] +
         (int32_t)sharedSrc[sharedBase + 2 * sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + 2 * sharedWidth]) /
            4,  // Red is is 4-tap 45° rotated (x)
        ((int32_t)sharedSrc[sharedBase + 1] + (int32_t)sharedSrc[sharedBase + sharedWidth] +
         (int32_t)sharedSrc[sharedBase + 2 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
            4,                                    // Green is 4-tap straight (+)
        sharedSrc[sharedBase + sharedWidth + 1],  // Blue is given
        255);
  }
}

/**
 * This kernel converts the buffer from Bayer-filtered BGGR to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * Uses bilinear interpolation within color planes.
 *
 * Each thread handles a 2*2 BGGR pixel block. The interpolation support is the 4*4 pixel block centered around the 2*2
 * block. Globally each thread block needs an extra pixel around itself.
 *
 */
__global__ void convertBayerBGGRToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                             unsigned width, unsigned height) {
  // x and y are the 2*2 block ids.
  const unsigned srcX = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned srcY = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  // Load the data to shared memory.
  extern __shared__ unsigned char sharedSrc[];
  const unsigned sharedWidth = loadBayerPattern(src, width, height, sharedSrc, srcX, srcY);
  __syncthreads();

  if (srcX < width && srcY < height) {
    const int sharedBase = sharedWidth + 1 + 2 * (sharedWidth * threadIdx.y + threadIdx.x);
    // Top-left component;
    dst[srcY * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase - 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 - sharedWidth] +
         (int32_t)sharedSrc[sharedBase - 1 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            4,  // Red is 4-tap 45° rotated (x)
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1] +
         (int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + sharedWidth]) /
            4,                  // Green is 4-tap straight (+)
        sharedSrc[sharedBase],  // Blue is given
        255);
    // Top-right component;
    dst[srcY * width + srcX + 1] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase + 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            2,                                                                      // Red is 2-tap vertical
        sharedSrc[sharedBase + 1],                                                  // Green is given
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2]) / 2,  // Blue is 2-tap horizontal
        255);
    // Bottom-left component;
    dst[(srcY + 1) * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase + sharedWidth - 1] + (int32_t)sharedSrc[sharedBase + sharedWidth + 1]) /
            2,                                // Red is 2-tap horizontal
        sharedSrc[sharedBase + sharedWidth],  // Green is given
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2 * sharedWidth]) /
            2,  // Blue is 2-tap vertical
        255);
    // Bottom-right component
    dst[(srcY + 1) * width + srcX + 1] = RGBA::pack(
        sharedSrc[sharedBase + sharedWidth + 1],  // Red is given
        ((int32_t)sharedSrc[sharedBase + 1] + (int32_t)sharedSrc[sharedBase + sharedWidth] +
         (int32_t)sharedSrc[sharedBase + 2 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
            4,  // Green is 4-tap straight (+)
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2] +
         (int32_t)sharedSrc[sharedBase + 2 * sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + 2 * sharedWidth]) /
            4,  // Blue is is 4-tap 45° rotated (x)
        255);
  }
}

/**
 * This kernel converts the buffer from Bayer-filtered GRBG to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * Uses bilinear interpolation within color planes.
 *
 * Each thread handles a 2*2 GRBG pixel block. The interpolation support is the 4*4 pixel block centered around the 2*2
 * block. Globally each thread block needs an extra pixel around itself.
 *
 */
__global__ void convertBayerGRBGToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                             unsigned width, unsigned height) {
  // x and y are the 2*2 block ids.
  const unsigned srcX = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned srcY = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  // Load the data to shared memory.
  extern __shared__ unsigned char sharedSrc[];
  const unsigned sharedWidth = loadBayerPattern(src, width, height, sharedSrc, srcX, srcY);
  __syncthreads();

  if (srcX < width && srcY < height) {
    const int sharedBase = sharedWidth + 1 + 2 * (sharedWidth * threadIdx.y + threadIdx.x);
    // Top-left component;
    dst[srcY * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1]) / 2,  // Red is 2-tap horizontal
        sharedSrc[sharedBase],                                                          // Green is given
        ((int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + sharedWidth]) /
            2,  // Blue is 2-tap vertical
        255);
    // Top-right component;
    dst[srcY * width + srcX + 1] = RGBA::pack(
        sharedSrc[sharedBase + 1],  // Red is given
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2] +
         (int32_t)sharedSrc[sharedBase + 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            4,  // Green is 4-tap straight (+)
        ((int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 - sharedWidth] +
         (int32_t)sharedSrc[sharedBase + sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + sharedWidth]) /
            4,  // Blue is 4-tap 45° rotated (x)
        255);
    // Bottom-left component;
    dst[(srcY + 1) * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1] +
         (int32_t)sharedSrc[sharedBase - 1 + 2 * sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
            4,  // Red is 4-tap 45° rotated (x)
        ((int32_t)sharedSrc[sharedBase - 1 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth] +
         (int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2 * sharedWidth]) /
            4,                                // Green is 4-tap straight (+)
        sharedSrc[sharedBase + sharedWidth],  // Blue is given
        255);
    // Bottom-right component
    dst[(srcY + 1) * width + srcX + 1] =
        RGBA::pack(((int32_t)sharedSrc[sharedBase + 1] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
                       2,                                    // Red is 2-tap vertical
                   sharedSrc[sharedBase + sharedWidth + 1],  // Green is given
                   ((int32_t)sharedSrc[sharedBase + sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + sharedWidth]) /
                       2,  // Blue is 2-tap horizontal
                   255);
  }
}

/**
 * This kernel converts the buffer from Bayer-filtered GBRG to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * Uses bilinear interpolation within color planes.
 *
 * Each thread handles a 2*2 GBRG pixel block. The interpolation support is the 4*4 pixel block centered around the 2*2
 * block. Globally each thread block needs an extra pixel around itself.
 *
 */
__global__ void convertBayerGBRGToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                             unsigned width, unsigned height) {
  // x and y are the 2*2 block ids.
  const unsigned srcX = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned srcY = 2 * (blockIdx.y * blockDim.y + threadIdx.y);

  // Load the data to shared memory.
  extern __shared__ unsigned char sharedSrc[];
  const unsigned sharedWidth = loadBayerPattern(src, width, height, sharedSrc, srcX, srcY);
  __syncthreads();

  if (srcX < width && srcY < height) {
    const int sharedBase = sharedWidth + 1 + 2 * (sharedWidth * threadIdx.y + threadIdx.x);
    // Top-left component;
    dst[srcY * width + srcX] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + sharedWidth]) /
            2,                                                                          // Red is 2-tap vertical
        sharedSrc[sharedBase],                                                          // Green is given
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1]) / 2,  // Blue is 2-tap horizontal
        255);
    // Top-right component;
    dst[srcY * width + srcX + 1] = RGBA::pack(
        ((int32_t)sharedSrc[sharedBase - sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 - sharedWidth] +
         (int32_t)sharedSrc[sharedBase + sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + sharedWidth]) /
            4,  // Red is 4-tap 45° rotated (x)
        ((int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2] +
         (int32_t)sharedSrc[sharedBase + 1 - sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth]) /
            4,                      // Green is 4-tap straight (+)
        sharedSrc[sharedBase + 1],  // Blue is given
        255);
    // Bottom-left component;
    dst[(srcY + 1) * width + srcX] = RGBA::pack(
        sharedSrc[sharedBase + sharedWidth],  // Red is given
        ((int32_t)sharedSrc[sharedBase - 1 + sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + sharedWidth] +
         (int32_t)sharedSrc[sharedBase] + (int32_t)sharedSrc[sharedBase + 2 * sharedWidth]) /
            4,  // Green is 4-tap straight (+)
        ((int32_t)sharedSrc[sharedBase - 1] + (int32_t)sharedSrc[sharedBase + 1] +
         (int32_t)sharedSrc[sharedBase - 1 + 2 * sharedWidth] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
            4,  // Blue is 4-tap 45° rotated (x)
        255);
    // Bottom-right component
    dst[(srcY + 1) * width + srcX + 1] =
        RGBA::pack(((int32_t)sharedSrc[sharedBase + sharedWidth] + (int32_t)sharedSrc[sharedBase + 2 + sharedWidth]) /
                       2,                                    // Red is 2-tap horizontal
                   sharedSrc[sharedBase + sharedWidth + 1],  // Green is given
                   ((int32_t)sharedSrc[sharedBase + 1] + (int32_t)sharedSrc[sharedBase + 1 + 2 * sharedWidth]) /
                       2,  // Blue is 2-tap vertical
                   255);
  }
}

/**
 * This kernel converts the buffer from BGR888 to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha)
 */
__global__ void convertBGRToRGBAKernel(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                       unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    dst[i] = RGBA::pack(src[3 * i + 2], src[3 * i + 1], src[3 * i], 0xff);
  }
}

/**
 * These kernels converts the buffer from various YUV422 representations to RGBA8888 out-of-place.
 * Pixels are all given full solidness (max alpha).
 */
__global__ void convertUYVYToRGBAKernel(cudaSurfaceObject_t dst, const unsigned char* __restrict__ src, unsigned width,
                                        unsigned height) {
  // each thread is responsible for a 2x1 pixel group
  // Two bytes per pixel. Y0 U Y1 V
  // Read 2x (y0), 2x+1 (u), 2x+2 (y1) 2x+3 (v)
  // Write x, x+1
  // Repeat for every line
  const unsigned pitch = width * 2;

  const unsigned x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const unsigned char u = src[y * pitch + 2 * x];  // Two bytes per pixel. U Y0 V Y1
    const unsigned char y0 = src[y * pitch + 2 * x + 1];
    const unsigned char v = src[y * pitch + 2 * x + 2];
    const unsigned char y1 = src[y * pitch + 2 * x + 3];
    const RGBDiff rgbDiff(yuv444ToRGBDiff(u, v));
    surf2Dwrite(YRGBDiffToRGBA(y0, rgbDiff), dst, x * 4, y);
    surf2Dwrite(YRGBDiffToRGBA(y1, rgbDiff), dst, (x + 1) * 4, y);
  }
}

__global__ void convertYUY2ToRGBAKernel(cudaSurfaceObject_t dst, const unsigned char* src, unsigned width,
                                        unsigned height) {
  // each thread is responsible for a 2x1 pixel group
  // Two bytes per pixel. Y0 U Y1 V
  // Read 2x (y0), 2x+1 (u), 2x+2 (y1) 2x+3 (v)
  // Write x, x+1
  // Repeat for every line
  const unsigned pitch = width * 2;

  const unsigned x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const unsigned char y0 = src[y * pitch + 2 * x];  // Two bytes per pixel. Y0 U Y1 V
    const unsigned char u = src[y * pitch + 2 * x + 1];
    const unsigned char y1 = src[y * pitch + 2 * x + 2];
    const unsigned char v = src[y * pitch + 2 * x + 3];
    const RGBDiff rgbDiff(yuv444ToRGBDiff(u, v));
    surf2Dwrite(YRGBDiffToRGBA(y0, rgbDiff), dst, x * 4, y);
    surf2Dwrite(YRGBDiffToRGBA(y1, rgbDiff), dst, (x + 1) * 4, y);
  }
}

/**
 * This kernel converts the buffer from 10 bits planar YUV422 to packed RGBA8888 out-of-place.
 * Each thread manages 2 pixels.
 * 10 bits values are padded to 16 bits, and are clamped to 8 bits during conversion
 * All pixels are solid.
 */
__global__ void convertYUV422P10ToRGBAKernel(cudaSurfaceObject_t dst, const uint16_t* src, unsigned width,
                                             unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  const uint16_t* uSrc = src + width * height;
  const uint16_t* vSrc = uSrc + width * height / 2;

  if (x < width / 2 && y < height) {
    uint32_t y0 = src[y * width + 2 * x] >> 2;
    uint32_t y1 = src[y * width + 2 * x + 1] >> 2;
    uint32_t u = uSrc[y * (width / 2) + x] >> 2;
    uint32_t v = vSrc[y * (width / 2) + x] >> 2;
    const RGBDiff rgbDiff = yuv444ToRGBDiff(u, v);
    surf2Dwrite(YRGBDiffToRGBA(y0, rgbDiff), dst, (2 * x) * 4, y);
    surf2Dwrite(YRGBDiffToRGBA(y1, rgbDiff), dst, (2 * x + 1) * 4, y);
  }
}

/**
 * This kernel converts the buffer from planar 12 bits 4:2:0 (YV12) to packed RGBA8888 out-of-place.
 * All pixels are solid.
 */
__global__ void convertYV12ToRGBAKernel(cudaSurfaceObject_t dst, const unsigned char* src, unsigned width,
                                        unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned sy = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned char* uSrc = src + width * height;
  const unsigned char* vSrc = uSrc + (width * height) / 4;

  if (sx < width / 2 && sy < height / 2) {
    const RGBDiff rgbDiff(yuv444ToRGBDiff(uSrc[sy * (width / 2) + sx], vSrc[sy * (width / 2) + sx]));
    surf2Dwrite(YRGBDiffToRGBA(src[(2 * sy) * width + 2 * sx], rgbDiff), dst, (2 * sx) * 4, 2 * sy);
    surf2Dwrite(YRGBDiffToRGBA(src[(2 * sy) * width + 2 * sx + 1], rgbDiff), dst, (2 * sx + 1) * 4, 2 * sy);
    surf2Dwrite(YRGBDiffToRGBA(src[(2 * sy + 1) * width + 2 * sx], rgbDiff), dst, (2 * sx) * 4, 2 * sy + 1);
    surf2Dwrite(YRGBDiffToRGBA(src[(2 * sy + 1) * width + 2 * sx + 1], rgbDiff), dst, (2 * sx + 1) * 4, 2 * sy + 1);
  }
}

__global__ void convertKernelGrayscale(uint32_t* __restrict__ dst, const unsigned char* __restrict__ src,
                                       unsigned width, unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    dst[i] = RGBA::pack((uint32_t)src[i], (uint32_t)src[i], (uint32_t)src[i], 0xff);
  }
}

__global__ void convertGrayscaleKernel(cudaSurfaceObject_t dst, const unsigned char* __restrict__ src, unsigned width,
                                       unsigned height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    unsigned i = y * width + x;
    surf2Dwrite(RGBA::pack((uint32_t)src[i], (uint32_t)src[i], (uint32_t)src[i], 0xff), dst, x * 4, y);
  }
}

}  // namespace Image
}  // namespace VideoStitch
