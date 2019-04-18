// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PLATFORM_INDEPENDENT_OPENCL_H_
#define PLATFORM_INDEPENDENT_OPENCL_H_

// clCreateCommandQueue is deprecated in OpenCL 2.0
// we are targeting OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#endif
