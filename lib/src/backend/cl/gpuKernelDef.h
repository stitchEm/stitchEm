// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Common definitions of functions that can be used in .gpu files
// to be shared between CUDA and OpenCL implementations
//
// ### OPENCL BACKEND ###
//

#pragma once

#include "../common/gpuKernelDef.h"

// --------------------------------------------
// thread/warp ID
// --------------------------------------------

static inline unsigned get_global_id_x() { return (unsigned)get_global_id(0); }

static inline unsigned get_global_id_y() { return (unsigned)get_global_id(1); }

// --------------------------------------------
// kernel definition
// --------------------------------------------

// a device kernel callable by the host
#define __global__ kernel

// a device kernel callable by the device
#define __device__ static inline

// --------------------------------------------
// address space
// --------------------------------------------

// note: __global__ is CUDA kernel type specifier
#define global_mem global

#define __restrict__ restrict

// --------------------------------------------
// data types
// --------------------------------------------

// char is 8 Bit in the OpenCL spec
typedef uchar uint8_t;

typedef ushort uint16_t;

// int is 32 Bit in the OpenCL spec
typedef int int32_t;
typedef uint uint32_t;

typedef uint4 color_t;

#ifdef CL_ARGS_WORKAROUND
#define GPU_CL_ARGS_WORKAROUND
// Redefining GPU functions to support OpenCL limitation in parameters format
#endif

// can't be a typedef because of the access qualifier
// [OpenCL] [CL_DEVICE_NOT_AVAILABLE] : OpenCL Error : Error: Build Program driver returned (518)
#define surface_t write_only image2d_t

#define Image_clamp8 clamp8

#ifndef make_int2
#define make_int2(A, B) (int2)((A), (B))
#endif  // make_int2

#ifndef make_float2
#define make_float2(A, B) (float2)((A), (B))
#endif  // make_float2

// --------------------------------------------
// utility functions
// --------------------------------------------

#define clamp_vs clamp

// No assertions in OpenCL
#define assert(expression)
