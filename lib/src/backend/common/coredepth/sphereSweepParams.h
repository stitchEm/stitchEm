// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/core/types.hpp"

// TODODEPTH the input textures are currently passed hardcoded as we don't support
// a variable length array of surfaces yet in the GPU backends
#define NUM_INPUTS 6

// 171013 GTX 780M: ~1 sec
#define SWEEP_ULTRA_FAST 1
// 171013 GTX 780M: ~4 sec
#define SWEEP_FAST 2
// 171013 GTX 780M: ~10 sec
#define SWEEP_MEDIUM 3
// 171013 GTX 780M: ~1 min
#define SWEEP_PLACEBO 4

// select sphere sweep preset to use
#define SWEEP_ALGO_PRESET SWEEP_MEDIUM

struct InputParams {
  float2 scale;
  float2 centerShift;
  vsDistortion distortion;
  vsfloat3x4 transform;
  vsfloat3x4 inverseTransform;
  int texWidth;
  int texHeight;
  int cropLeft;
  int cropRight;
  int cropTop;
  int cropBottom;
};

struct InputParams6 {
  struct InputParams params[6];
};

#if SWEEP_ALGO_PRESET == SWEEP_ULTRA_FAST
static const int numSphereScales = 16;
static const int numSphereScalesRefine = 5;
static const int patchWidth = 1;
static const int patchHeight = 1;
static const int patchWidthStep = 1;
static const int patchHeightStep = 1;

#elif SWEEP_ALGO_PRESET == SWEEP_FAST
static const int numSphereScales = 32;
static const int numSphereScalesRefine = 9;
static const int patchWidth = 5;
static const int patchHeight = 5;
static const int patchWidthStep = 3;
static const int patchHeightStep = 3;

#elif SWEEP_ALGO_PRESET == SWEEP_MEDIUM
static const int numSphereScales = 32;
static const int numSphereScalesRefine = 17;
static const int patchWidth = 7;
static const int patchHeight = 7;
static const int patchWidthStep = 5;
static const int patchHeightStep = 5;

#elif SWEEP_ALGO_PRESET == SWEEP_PLACEBO
static const int numSphereScales = 128;
static const int numSphereScalesRefine = 33;
static const int patchWidth = 11;
static const int patchHeight = 11;
static const int patchWidthStep = 7;
static const int patchHeightStep = 7;

#else
#error Select a valid SWEEP preset
#endif

static const int patchSize = patchWidth * patchHeight;
static const int patchSizeStep = patchWidthStep * patchHeightStep;
