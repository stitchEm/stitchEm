// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../core1/warpKernelDef.h"

#define sync_threads barrier

#define sqrt4 sqrt
#define fmaxf fmax
#define nullptr 0
#define __globalmem__ __global

#include "backend/common/coredepth/sphereSweepParams.h"

// workaround for Intel OpenCL compiler on Mac Mini
// if the argument size is too big, the compiler has an internal error
#define INPUT_PARAMS_T constant const struct InputParams6*

inline struct InputParams6 get_input_params(INPUT_PARAMS_T in) { return *in; }

static __constant sampler_t depthSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

static inline float read_depth_vs(read_only image2d_t tex, float2 uv) { return read_imagef(tex, depthSampler, uv).x; }

#include "backend/common/coredepth/sphereSweep.gpu"
