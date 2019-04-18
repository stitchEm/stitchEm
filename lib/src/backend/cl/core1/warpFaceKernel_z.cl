// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "warpKernelDef.h"

// #include "backend/common/core1/warpFaceKernel.gpu"

// don't have too many kernel definitions (many template parameters!) in a single file
// or the GPU compiler runs out of heap or randomly dies (both AMD OpenCL and Nvidia CUDA)

#define WARPKERNEL_VARIANT_INCLUDE "warpFaceKernel_face_z.gpu.incl"
#define WARPFACEKERNEL_VARIANT_INCLUDE "warpFaceKernel_PhotoCorrectionT.gpu.incl"
#include "backend/common/core1/warpKernelSphere_distortion.gpu"
#undef WARPFACEKERNEL_VARIANT_INCLUDE
#undef WARPKERNEL_VARIANT_INCLUDE
