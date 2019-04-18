// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#define WARPKERNEL_VARIANT_INCLUDE "warpFaceKernel_face_x.gpu.incl"
#define WARPFACEKERNEL_VARIANT_INCLUDE "warpFaceKernel_PhotoCorrectionT.gpu.incl"
#include "backend/common/core1/warpKernelSphere_distortion.gpu"
#undef WARPFACEKERNEL_VARIANT_INCLUDE
#undef WARPKERNEL_VARIANT_INCLUDE

#define WARPKERNEL_VARIANT_INCLUDE "warpFaceKernel_face_y.gpu.incl"
#define WARPFACEKERNEL_VARIANT_INCLUDE "warpFaceKernel_PhotoCorrectionT.gpu.incl"
#include "backend/common/core1/warpKernelSphere_distortion.gpu"
#undef WARPFACEKERNEL_VARIANT_INCLUDE
#undef WARPKERNEL_VARIANT_INCLUDE

#define WARPKERNEL_VARIANT_INCLUDE "warpFaceKernel_face_z.gpu.incl"
#define WARPFACEKERNEL_VARIANT_INCLUDE "warpFaceKernel_PhotoCorrectionT.gpu.incl"
#include "backend/common/core1/warpKernelSphere_distortion.gpu"
#undef WARPFACEKERNEL_VARIANT_INCLUDE
#undef WARPKERNEL_VARIANT_INCLUDE
