// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#if defined(VS_OPENCL)

// TODO: use OpenCL API types, e.g. cl_float2?

typedef struct {
  float x;
  float y;
} float2;
typedef struct {
  double x;
  double y;
} double2;
typedef struct {
  int x;
  int y;
} int2;

typedef struct {
  float x;
  float y;
  float z;
} float3;
typedef struct {
  double x;
  double y;
  double z;
} double3;
typedef struct {
  char x;
  char y;
  char z;
} char3;
typedef struct {
  unsigned char x;
  unsigned char y;
  unsigned char z;
} uchar3;

typedef struct {
  float x;
  float y;
  float z;
  float w;
} float4;

float2 make_float2(float x, float y);
int2 make_int2(int x, int y);
double2 make_double2(double x, double y);

float3 make_float3(float x, float y, float z);
double3 make_double3(double x, double y, double z);

float4 make_float4(float x, float y, float z, float w);

#elif defined(CL_VERSION_1_2)

// nothing to do

#else

// include functions from CUDA
// TODO: are they 100% compatible, or does this approach need to be changed?
// what about alignment?

// if your build breaks here it's because you are missing the CUDA include folder
// solution: install CUDA SDK or copy the CUDA headers from a CUDA SDK installation on another machine to an include
// folder
#include <vector_functions.h>

#endif
