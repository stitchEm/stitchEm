// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/logging.hpp"
#include "libvideostitch/status.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_ERROR(stmt) VideoStitch::Cuda::cudaCheckError((stmt), __FILE__, __LINE__)
#define CUDA_STATUS VideoStitch::Cuda::cudaCheckStatus(__FILE__, __LINE__)

namespace VideoStitch {
namespace Cuda {

/**
 * Prints error messages
 * @param err The return value of a CUDA function.
 * @param file Filename.
 * @param line Line number.
 * @note Usually not called directly; use CUDA_ERROR;
 */
Status cudaCheckError(cudaError err, const char *file, int line);

/**
 * Prints error messages
 * @param err The return value of a CUDA function.
 * @param file Filename.
 * @param line Line number.
 * @note Usually not called directly; use CUDA_ERROR;
 */
Status cudaCheckError(CUresult err, const char *file, int line);

/*
 * CUDA kernel calls do not return an error code or status
 * This can be used to see whether there are errors in the CUDA runtime
 * Usually not called directly; use CUDA_STATUS;
 */
Status cudaCheckStatus(const char *file, int line);

}  // namespace Cuda
}  // namespace VideoStitch
