// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

typedef struct {
  float values[2][2];
} vsfloat2x2;

#ifndef GPU_CL_ARGS_WORKAROUND
typedef struct {
  float values[3][3];
#ifdef CL_ARGS_WORKAROUND
  /* need to align kernel argument size between CPU & GPU */
  float dummyPadding[7];
#endif
} vsfloat3x3;

typedef struct {
  float values[3][4];
#ifdef CL_ARGS_WORKAROUND
  float dummyPadding[4];
#endif
} vsfloat3x4;

typedef struct {
  float values[11];
  vsfloat3x3 scheimpflugForward;
  vsfloat3x3 scheimpflugInverse;
  char distortionBitFlag;
  char dummyPadding[3];
} vsDistortion;
#else
#define vsfloat3x3 float16
#define vsfloat3x4 float16
#define vsDistortion float16
#endif

typedef struct {
  int left;
  int right;
  int top;
  int bottom;
} RectInt;

typedef struct {
  int width;
  int height;
} SizeInt;

typedef struct vsQuarticSolution {
  int solutionCount;
  float x[4];
} vsQuarticSolution;

#define TANGENTIAL_DISTORTION_BIT 0x01
#define THIN_PRISM_DISTORTION_BIT 0x02
#define SCHEIMPFLUG_DISTORTION_BIT 0x04

#define LARGE_ROOT 1000.0f
#define INVALID_INVERSE_DISTORTION 100000.0f
#define INVALID_FLOW_VALUE -100000.0f
