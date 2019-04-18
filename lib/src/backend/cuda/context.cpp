// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/context.hpp"
#include "cuda/error.hpp"
#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/allocator.hpp"
#include <cuda_runtime.h>

#ifndef TEST_KERNEL_PROGRESS_BAR
#define TEST_KERNEL_PROGRESS_BAR 0
#endif

#if TEST_KERNEL_PROGRESS_BAR
#include <thread>
#include <chrono>
#endif

namespace VideoStitch {
namespace GPU {
namespace Context {

Status compileAllKernels(Util::Algorithm::ProgressReporter* /*pReporter*/) {
  // TODO trigger optimization for target architecture, if binary not available?
  return Status::OK();
}

Status compileAllKernelsOnSelectedDevice(int, const bool, Util::Algorithm::ProgressReporter*) {
#if TEST_KERNEL_PROGRESS_BAR
  FAIL_CONDITION(!pReporter->notify("Compiling kernels", 0.0), VideoStitch::Origin::GPU,
                 VideoStitch::ErrType::RuntimeError, "Canceled");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  FAIL_CONDITION(!pReporter->notify("Compiling kernels", 0.3), VideoStitch::Origin::GPU,
                 VideoStitch::ErrType::RuntimeError, "Canceled");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  FAIL_CONDITION(!pReporter->notify("Compiling kernels", 0.5), VideoStitch::Origin::GPU,
                 VideoStitch::ErrType::RuntimeError, "Canceled");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  FAIL_CONDITION(!pReporter->notify("Compiling kernels", 0.8), VideoStitch::Origin::GPU,
                 VideoStitch::ErrType::RuntimeError, "Canceled");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  pReporter->notify("Compiling kernels", 1.0);
#endif
  // return OK for cuda
  return Status::OK();
}

Status destroy() { return CUDA_ERROR(cudaDeviceReset()); }

Status setDefaultBackendDeviceAndCheck(const int vsDeviceID) {
  FAIL_RETURN(setDefaultBackendDeviceVS(vsDeviceID));
  return Core::OpenGLAllocator::createSourceSurface(1, 1).status();
}

}  // namespace Context
}  // namespace GPU
}  // namespace VideoStitch
