// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Common definitions of functions that can be used in .gpu files
// to be shared between CUDA and OpenCL implementations
//
// ### CUDA BACKEND ###
//

#pragma once

#include "../common/gpuKernelDef.h"

// --------------------------------------------
// thread/warp ID
// --------------------------------------------

inline __device__ unsigned get_global_id_x() { return blockIdx.x * blockDim.x + threadIdx.x; }

inline __device__ unsigned get_global_id_y() { return blockIdx.y * blockDim.y + threadIdx.y; }

// --------------------------------------------
// address space
// --------------------------------------------

#define global_mem

#define read_only
#define write_only

// --------------------------------------------
// data types
// --------------------------------------------

typedef cudaSurfaceObject_t surface_t;
typedef uint32_t color_t;

// --------------------------------------------
// utility functions
// --------------------------------------------

inline __device__ __host__ uint32_t clamp_vs(uint32_t v, uint32_t minVal, uint32_t maxVal) {
  return v < minVal ? minVal : (v > maxVal ? maxVal : v);
}

inline __device__ void surface_read(uint32_t *color, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dread(color, surf, x * sizeof(uint32_t), y);
}

inline __device__ void surface_read_depth(float *depth, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dread(depth, surf, x * sizeof(float), y, cudaBoundaryModeClamp);
}

inline __device__ void surface_write_i(uint32_t color, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dwrite(color, surf, x * sizeof(uint32_t), y);
}

inline __device__ void surface_write(uint32_t color, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dwrite(color, surf, x * sizeof(uint32_t), y);
}

inline __device__ void surface_write_depth(float depth, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dwrite(depth, surf, x * sizeof(depth), y);
}

inline __device__ void surface_write_f(float4 color, cudaSurfaceObject_t surf, int x, int y) {
  uchar4 uc;
  uc.x = fminf(255.0f, color.x * 255.0f);
  uc.y = fminf(255.0f, color.y * 255.0f);
  uc.z = fminf(255.0f, color.z * 255.0f);
  uc.w = fminf(255.0f, color.w * 255.0f);
  surf2Dwrite(uc, surf, x * sizeof(uc), y);
}

inline __device__ void coord_write(float2 coord, cudaSurfaceObject_t surf, int x, int y) {
  surf2Dwrite(coord, surf, x * sizeof(float2), y);
}

// --------------------------------------------
// pixel packing/unpacking
// --------------------------------------------

#define Image_RGB210_a VideoStitch::Image::RGB210::a
#define Image_RGB210_r VideoStitch::Image::RGB210::r
#define Image_RGB210_g VideoStitch::Image::RGB210::g
#define Image_RGB210_b VideoStitch::Image::RGB210::b
#define Image_RGB210_pack VideoStitch::Image::RGB210::pack

#define Image_RGBA_a VideoStitch::Image::RGBA::a
#define Image_RGBA_r VideoStitch::Image::RGBA::r
#define Image_RGBA_g VideoStitch::Image::RGBA::g
#define Image_RGBA_b VideoStitch::Image::RGBA::b
#define Image_RGBA_pack VideoStitch::Image::RGBA::pack

#define Image_clamp8 VideoStitch::Image::clamp8
