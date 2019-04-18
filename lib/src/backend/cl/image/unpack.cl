// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../gpuKernelDef.h"

#include "imageFormat.h"

// change from CUDA: RGBDiff is value, not reference
static inline float4 YRGBDiffToRGBA(unsigned char y, const int3 rgbDiff) {
  const int ya = (1192 * (y - 16)) >> 10;
  return (float4){clamp8(ya + rgbDiff.x) / 255.f, clamp8(ya + rgbDiff.y) / 255.f, clamp8(ya + rgbDiff.z) / 255.f, 1.f};
}

#define nv12_surface_write surface_write_f

#include "backend/common/image/unpack.gpu"

static __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unpackKernelRGB(global unsigned char* dst, unsigned pitch, global const unsigned int* src, unsigned width,
                            unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    unsigned int val = src[y * width + x];
    dst[y * pitch + 3 * x] = (unsigned char)Image_RGBA_r(val);
    dst[y * pitch + 3 * x + 1] = (unsigned char)Image_RGBA_g(val);
    dst[y * pitch + 3 * x + 2] = (unsigned char)Image_RGBA_b(val);
  }
}

kernel void unpackKernelRGBSource(global unsigned char* dst, unsigned pitch, read_only image2d_t src, unsigned width,
                                  unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    float4 val = read_imagef(src, sampler, (int2)(x, y));
    dst[y * pitch + 3 * x] = (unsigned char)(val.x * 255.f);
    dst[y * pitch + 3 * x + 1] = (unsigned char)(val.y * 255.f);
    dst[y * pitch + 3 * x + 2] = (unsigned char)(val.z * 255.f);
  }
}

kernel void unpackKernelGrayscaleSource(global unsigned char* dst, unsigned pitch, read_only image2d_t src,
                                        unsigned width, unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    float4 px = read_imagef(src, sampler, (int2)(x, y));
    dst[y * pitch + x] = (unsigned char)clamp8((int)(65.481f * px.x + 128.553f * px.y + 24.966f * px.z + 16.5f));
  }
}

kernel void unpackKernelRGBA(global unsigned char* dst, unsigned pitch, read_only image2d_t src, unsigned width,
                             unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    float4 val = read_imagef(src, sampler, (int2)(x, y));
    dst[y * pitch + 4 * x] = (unsigned char)(val.x * 255.f);
    dst[y * pitch + 4 * x + 1] = (unsigned char)(val.y * 255.f);
    dst[y * pitch + 4 * x + 2] = (unsigned char)(val.z * 255.f);
    dst[y * pitch + 4 * x + 3] = (unsigned char)(val.w * 255.f);
  }
}

kernel void unpackKernelF32C1(global float* dst, unsigned pitch, read_only image2d_t src, unsigned width,
                              unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    float4 val = read_imagef(src, sampler, (int2)(x, y));
    dst[y * pitch + x] = val.x;
  }
}

kernel void unpackKernelDepthSource(global unsigned char* yDst, unsigned yPitch, global unsigned char* uDst,
                                    unsigned uPitch, global unsigned char* vDst, unsigned vPitch,
                                    read_only image2d_t src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned x = 2 * (unsigned)get_global_id(0);
  unsigned y = 2 * (unsigned)get_global_id(1);

  int u = 0;
  int v = 0;

  if (x < width && y < height) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        float4 depth = read_imagef(src, sampler, (int2)(x + i, y + j));
        // convert to millimeters and truncate
        unsigned int val = (unsigned int)min(convert_uint_sat_rtn(depth.x * 1000.f), (unsigned int)65279);
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
    uDst[(y * uPitch + x) / 2] = (unsigned char)((u + 2) / 4);
    vDst[(y * vPitch + x) / 2] = (unsigned char)((v + 2) / 4);
  }
}

kernel void unpackKernelDepth(global unsigned char* yDst, unsigned yPitch, global unsigned char* uDst, unsigned uPitch,
                              global unsigned char* vDst, unsigned vPitch, global const float* src, unsigned width,
                              unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned x = 2 * (unsigned)get_global_id(0);
  unsigned y = 2 * (unsigned)get_global_id(1);

  int u = 0;
  int v = 0;

  if (x < width && y < height) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        float depth = src[(y + j) * width + x + i];
        // convert to millimeters and truncate
        unsigned int val = (unsigned int)min(convert_uint_sat_rtn(depth * 1000.f), (unsigned int)65279);
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
    uDst[(y * uPitch + x) / 2] = (unsigned char)((u + 2) / 4);
    vDst[(y * vPitch + x) / 2] = (unsigned char)((v + 2) / 4);
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
kernel void unpackKernelYV12(global unsigned char* yDst, unsigned yPitch, global unsigned char* uDst, unsigned uPitch,
                             global unsigned char* vDst, unsigned vPitch, global const unsigned int* src,
                             unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (unsigned)get_global_id(0);
  unsigned sy = 2 * (unsigned)get_global_id(1);
  int u = 0;
  int v = 0;

  if (sx < width && sy < height) {
    {
      unsigned int val = src[sy * width + sx];
      int r = Image_RGBA_r(val);
      int g = Image_RGBA_g(val);
      int b = Image_RGBA_b(val);
      int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = (unsigned char)y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        unsigned int val = src[sy * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        unsigned int val = src[(sy + 1) * width + sx];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        unsigned int val = src[(sy + 1) * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uDst[(sy * uPitch + sx) / 2] = (unsigned char)(u / 4);
      vDst[(sy * vPitch + sx) / 2] = (unsigned char)(v / 4);
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        unsigned int val = src[sy * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = (unsigned char)(u / 2);
        vDst[(sy * vPitch + sx) / 2] = (unsigned char)(v / 2);
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (sx == (width - 1) && sy + 1 < height) {
    unsigned int val = src[(sy + 1) * width + sx];
    int r = Image_RGBA_r(val);
    int g = Image_RGBA_g(val);
    int b = Image_RGBA_b(val);
    int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    yDst[(sy + 1) * width + sx] = (unsigned char)y;
    u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    uDst[(sy * ((width + 1) / 2) + sx) / 2] = (unsigned char)(u / 2);
    vDst[(sy * ((width + 1) / 2) + sx) / 2] = (unsigned char)(v / 2);
  }
}

kernel void unpackKernelYV12Source(global unsigned char* yDst, unsigned yPitch, global unsigned char* uDst,
                                   unsigned uPitch, global unsigned char* vDst, unsigned vPitch,
                                   read_only image2d_t src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (unsigned)get_global_id(0);
  unsigned sy = 2 * (unsigned)get_global_id(1);
  int u = 0;
  int v = 0;

  if (sx < width && sy < height) {
    {
      float4 val = read_imagef(src, sampler, (int2)(sx, sy));
      int r = (int)(val.x * 255.f);
      int g = (int)(val.y * 255.f);
      int b = (int)(val.z * 255.f);
      int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = (unsigned char)y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        float4 val = read_imagef(src, sampler, (int2)(sx, sy + 1));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy + 1));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uDst[(sy * uPitch + sx) / 2] = (unsigned char)(u / 4);
      vDst[(sy * vPitch + sx) / 2] = (unsigned char)(v / 4);
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uDst[(sy * uPitch + sx) / 2] = (unsigned char)(u / 2);
        vDst[(sy * vPitch + sx) / 2] = (unsigned char)(v / 2);
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (sx == (width - 1) && sy + 1 < height) {
    float4 val = read_imagef(src, sampler, (int2)(sx, sy + 1));
    int r = (int)(val.x * 255.f);
    int g = (int)(val.y * 255.f);
    int b = (int)(val.z * 255.f);
    int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
    u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    uDst[(sy * uPitch + sx) / 2] = (unsigned char)(u / 2);
    vDst[(sy * vPitch + sx) / 2] = (unsigned char)(v / 2);
  }
}

/**
 * This kernel converts the buffer from RGBA to interleaved 12 bits 4:2:0 (NV12) out-of-place.
 * The conversion is undefined for pixels with 0 alpha.
 *
 * Y0 Y1 Y2 Y3
 * ...
 * U0 V0 U1 V1
 * ...
 */
kernel void unpackKernelNV12(global unsigned char* yDst, unsigned yPitch, global unsigned char* uvDst, unsigned uvPitch,
                             global const unsigned int* src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (unsigned)get_global_id(0);
  unsigned sy = 2 * (unsigned)get_global_id(1);
  int u = 0;
  int v = 0;

  if (sx < width && sy < height) {
    {
      unsigned int val = src[sy * width + sx];
      int r = Image_RGBA_r(val);
      int g = Image_RGBA_g(val);
      int b = Image_RGBA_b(val);
      int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = (unsigned char)y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        unsigned int val = src[sy * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        unsigned int val = src[(sy + 1) * width + sx];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        unsigned int val = src[(sy + 1) * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uvDst[(sy * uvPitch / 2) + sx] = (unsigned char)(u / 4);
      uvDst[(sy * uvPitch / 2) + sx + 1] = (unsigned char)(v / 4);
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        unsigned int val = src[sy * width + sx + 1];
        int r = Image_RGBA_r(val);
        int g = Image_RGBA_g(val);
        int b = Image_RGBA_b(val);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * width) / 2 + sx] = (unsigned char)(u / 2);
        uvDst[(sy * width) / 2 + sx + 1] = (unsigned char)(v / 2);
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (sx == (width - 1) && sy + 1 < height) {
    unsigned int val = src[(sy + 1) * width + sx];
    int r = Image_RGBA_r(val);
    int g = Image_RGBA_g(val);
    int b = Image_RGBA_b(val);
    int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    yDst[(sy + 1) * width + sx] = (unsigned char)y;
    u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    uvDst[(sy * width) / 2 + sx] = (unsigned char)(u / 2);
    uvDst[(sy * width) / 2 + sx + 1] = (unsigned char)(v / 2);
  }
}

kernel void unpackKernelNV12Source(global unsigned char* yDst, unsigned yPitch, global unsigned char* uvDst,
                                   unsigned uvPitch, read_only image2d_t src, unsigned width, unsigned height) {
  // each thread is responsible for a 2x2 pixel group
  unsigned sx = 2 * (unsigned)get_global_id(0);
  unsigned sy = 2 * (unsigned)get_global_id(1);
  int u = 0;
  int v = 0;
  if (sx < width && sy < height) {
    {
      float4 val = read_imagef(src, sampler, (int2)(sx, sy));
      int r = (int)(val.x * 255.f);
      int g = (int)(val.y * 255.f);
      int b = (int)(val.z * 255.f);
      int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      yDst[sy * yPitch + sx] = (unsigned char)y;
      u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
      v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    }
    if (sx + 1 < width && sy + 1 < height) {
      // general case
      {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        float4 val = read_imagef(src, sampler, (int2)(sx, sy + 1));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy + 1));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[(sy + 1) * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
      }
      uvDst[(sy * uvPitch / 2) + sx] = (unsigned char)(u / 4);
      uvDst[(sy * uvPitch / 2) + sx + 1] = (unsigned char)(v / 4);
    } else {
      // border case with odd width / height
      if (sx + 1 < width) {
        float4 val = read_imagef(src, sampler, (int2)(sx + 1, sy));
        int r = (int)(val.x * 255.f);
        int g = (int)(val.y * 255.f);
        int b = (int)(val.z * 255.f);
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        yDst[sy * yPitch + sx + 1] = (unsigned char)y;
        u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        uvDst[(sy * width) / 2 + sx] = (unsigned char)(u / 2);
        uvDst[(sy * width) / 2 + sx + 1] = (unsigned char)(v / 2);
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (sx == (width - 1) && sy + 1 < height) {
    float4 val = read_imagef(src, sampler, (int2)(sx, sy + 1));
    int r = (int)(val.x * 255.f);
    int g = (int)(val.y * 255.f);
    int b = (int)(val.z * 255.f);
    int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    yDst[(sy + 1) * yPitch + sx] = (unsigned char)y;
    u += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    v += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    uvDst[(sy * uvPitch) / 2 + sx] = (unsigned char)(u / 2);
    uvDst[(sy * uvPitch) / 2 + sx + 1] = (unsigned char)(v / 2);
  }
}

/**
 * This kernel converts the buffer from RGBA to 10 bits planar YUV422 out-of-place.
 * Pixels are all given full solidness (max alpha).
 * 10 bits values are padded to 16 bits.
 */

kernel void unpackYUV422P10Kernel(global unsigned short* yDst, unsigned yPitch, global unsigned short* uDst,
                                  unsigned uPitch, global unsigned short* vDst, unsigned vPitch,
                                  global const unsigned int* src, unsigned width, unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);

  if (x < width / 2 && y < height) {
    unsigned int val0 = src[y * width + 2 * x];
    int r0 = Image_RGBA_r(val0);
    int g0 = Image_RGBA_g(val0);
    int b0 = Image_RGBA_b(val0);
    unsigned int val1 = src[y * width + 2 * x + 1];
    int r1 = Image_RGBA_r(val1);
    int g1 = Image_RGBA_g(val1);
    int b1 = Image_RGBA_b(val1);
    unsigned int u = 0, v = 0;
    int y0 = (((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16) << 2;
    int y1 = (((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16) << 2;
    u += (((-38 * r0 - 74 * g0 + 112 * b0 + 128) >> 8) + 128) << 2;
    u += (((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128) << 2;
    v += (((112 * r0 - 94 * g0 - 18 * b0 + 128) >> 8) + 128) << 2;
    v += (((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128) << 2;
    yDst[y * yPitch + 2 * x] = (unsigned short)y0;
    yDst[y * yPitch + 2 * x + 1] = (unsigned short)y1;
    uDst[y * uPitch + x] = (unsigned short)(u / 2);
    vDst[y * vPitch + x] = (unsigned short)(v / 2);
  }
}

/**
 * This kernel converts the buffer from planar 12 bits 4:2:0 (YV12) to packed RGBA8888 out-of-place.
 * All pixels are solid.
 */
kernel void convertYV12ToRGBAKernel(write_only image2d_t dst, global const unsigned char* src, unsigned width,
                                    unsigned height) {
  // each thread is responsible for a 2x2 pixel group

  const unsigned sx = (unsigned)get_global_id(0);
  const unsigned sy = (unsigned)get_global_id(1);

  global const unsigned char* uSrc = src + width * height;
  global const unsigned char* vSrc = uSrc + (width * height) / 4;

  if (sx < width / 2 && sy < height / 2) {
    const int3 rgbDiff = yuv444ToRGBDiff(uSrc[sy * (width / 2) + sx], vSrc[sy * (width / 2) + sx]);
    {
      int2 coords = {2 * sx, 2 * sy};
      write_imagef(dst, coords, YRGBDiffToRGBA(src[coords.x + coords.y * width], rgbDiff));
    }
    {
      int2 coords = {2 * sx + 1, 2 * sy};
      write_imagef(dst, coords, YRGBDiffToRGBA(src[coords.x + coords.y * width], rgbDiff));
    }
    {
      int2 coords = {2 * sx, 2 * sy + 1};
      write_imagef(dst, coords, YRGBDiffToRGBA(src[coords.x + coords.y * width], rgbDiff));
    }
    {
      int2 coords = {2 * sx + 1, 2 * sy + 1};
      write_imagef(dst, coords, YRGBDiffToRGBA(src[coords.x + coords.y * width], rgbDiff));
    }
  }
}

kernel void convertYUY2ToRGBAKernel(write_only image2d_t dst, global const unsigned char* src, unsigned width,
                                    unsigned height) {
  // each thread is responsible for a 2x1 pixel group
  // Two bytes per pixel. Y0 U Y1 V
  // Read 2x (y0), 2x+1 (u), 2x+2 (y1) 2x+3 (v)
  // Write x, x+1
  // Repeat for every line
  const unsigned pitch = width * 2;

  const unsigned x = 2 * (unsigned)get_global_id(0);
  const unsigned y = (unsigned)get_global_id(1);

  if (x < width && y < height) {
    const unsigned char y0 = src[y * pitch + 2 * x];  // Two bytes per pixel. Y0 U Y1 V
    const unsigned char u = src[y * pitch + 2 * x + 1];
    const unsigned char y1 = src[y * pitch + 2 * x + 2];
    const unsigned char v = src[y * pitch + 2 * x + 3];

    const int3 rgbDiff = yuv444ToRGBDiff(u, v);
    {
      int2 coords = {x, y};
      write_imagef(dst, coords, YRGBDiffToRGBA(y0, rgbDiff));
    }
    {
      int2 coords = {x + 1, y};
      write_imagef(dst, coords, YRGBDiffToRGBA(y1, rgbDiff));
    }
  }
}

kernel void convertRGB210ToRGBAKernel(write_only image2d_t dst, global const unsigned* src, unsigned width,
                                      unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    unsigned v = src[y * width + x];
    int2 coords = {x, y};
    write_imagef(dst, coords,
                 (float4)(clamp8(Image_RGB210_r(v)) / 255.f, clamp8(Image_RGB210_g(v)) / 255.f,
                          clamp8(Image_RGB210_b(v)) / 255.f, Image_RGB210_a(v) / 255.f));
  }
}

kernel void convertRGBToRGBAKernel(write_only image2d_t dst, global const unsigned char* src, unsigned width,
                                   unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);
  if (x < width && y < height) {
    write_imagef(
        dst, (int2)(x, y),
        (float4)(src[y * 3 * width + 3 * x], src[y * 3 * width + 3 * x + 1], src[y * 3 * width + 3 * x + 2], 255.f) /
            255.f);
  }
}

/**
 * This kernel converts the buffer from 10 bits planar YUV422 to packed RGBA8888 out-of-place.
 * Each thread manages 2 pixels.
 * 10 bits values are padded to 16 bits, and are clamped to 8 bits during conversion
 * All pixels are solid.
 */
kernel void convertYUV422P10ToRGBAKernel(write_only image2d_t dst, global const unsigned short* src, unsigned width,
                                         unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);

  global const unsigned short* uSrc = src + width * height;
  global const unsigned short* vSrc = uSrc + width * height / 2;

  if (x < width / 2 && y < height) {
    unsigned int y0 = (src[y * width + 2 * x]) >> 2;
    unsigned int y1 = (src[y * width + 2 * x + 1]) >> 2;
    unsigned int u = (uSrc[y * (width / 2) + x]) >> 2;
    unsigned int v = (vSrc[y * (width / 2) + x]) >> 2;
    const RGBDiff rgbDiff = yuv444ToRGBDiff((unsigned char)u, (unsigned char)v);
    write_imagef(dst, (int2)((2 * x), y), YRGBDiffToRGBA((unsigned char)y0, rgbDiff));
    write_imagef(dst, (int2)((2 * x + 1), y), YRGBDiffToRGBA((unsigned char)y1, rgbDiff));
  }
}

/**
 * This kernel converts the buffer from 8 bits monochrome Grayscale to packed RGBA8888 out-of-place.
 * Each thread manages 2 pixels.
 * All pixels are solid.
 */
// TODO Not tested
kernel void convertGrayscaleToRGBAKernel(write_only image2d_t dst, global const unsigned short* src, unsigned width,
                                         unsigned height) {
  unsigned x = (unsigned)get_global_id(0);
  unsigned y = (unsigned)get_global_id(1);

  if (x < width && y < height) {
    unsigned int lum = *(src + width * y + x);
    float4 rgbaVal = (float4)(lum, lum, lum, 1.0f);
    write_imagef(dst, (int2)((4 * x), y), rgbaVal);
  }
}
