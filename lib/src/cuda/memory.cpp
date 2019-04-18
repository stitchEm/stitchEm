// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "memory.hpp"

#include "libvideostitch/logging.hpp"
#include "backend/common/allocStats.hpp"

#include <cuda_runtime.h>

#include "error.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <mutex>

// If the following is defined, all cudaMallocXxxVS functions will bin memory allocation in named pools for memory
// debugging In that case, a global lock will be used which makes it possible to share pools between threads without
// caring about thread-safety.
#define USE_VS_MALLOC_POOLS

#undef max

namespace VideoStitch {
namespace Cuda {

#ifdef USE_VS_MALLOC_POOLS
namespace {
AllocStatsMap deviceStats("Device");
AllocStatsMap hostStats("Host");
}  // namespace
#endif  // USE_VS_MALLOC_POOLS

std::size_t getDevicePoolCurrentSize(void) {
#ifdef USE_VS_MALLOC_POOLS
  return deviceStats.bytesUsed();
#else
  return 0;
#endif
}

std::vector<std::size_t> getDevicePoolCurrentSizeByDevices(void) {
#ifdef USE_VS_MALLOC_POOLS
  return deviceStats.bytesUsedByDevices();
#else
  return std::vector<std::size_t>();
#endif
}

std::size_t getHostPoolCurrentSize(void) {
#ifdef USE_VS_MALLOC_POOLS
  return hostStats.bytesUsed();
#else
  return 0;
#endif
}

std::vector<std::size_t> getHostPoolCurrentSizeByDevices(void) {
#ifdef USE_VS_MALLOC_POOLS
  return hostStats.bytesUsedByDevices();
#else
  return std::vector<std::size_t>();
#endif
}

void printDevicePool() {
#ifdef USE_VS_MALLOC_POOLS
  deviceStats.print(std::cout);
#endif
}

void printHostPool() {
#ifdef USE_VS_MALLOC_POOLS
  hostStats.print(std::cout);
#endif
}

#ifndef NDEBUG
#define PRINT_FILELINE                                                                                 \
  if (file) {                                                                                          \
    Logger::get(VideoStitch::Logger::Error) << " (at " << file << ", l. " << line << ")" << std::endl; \
  };
#define FILELINE_ARGS const char *file, int line
#else
#define PRINT_FILELINE
#define FILELINE_ARGS const char* /*file*/, int /*line*/
#endif

Status __mallocVS(void** buf, size_t size, const char* name, unsigned /*flagsUnused*/, FILELINE_ARGS) {
  if (!CUDA_ERROR(cudaMalloc(buf, size)).ok()) {
    Logger::get(VideoStitch::Logger::Error) << "Could not allocate " << size << " bytes of GPU memory.";
    PRINT_FILELINE
    Logger::get(VideoStitch::Logger::Error) << std::endl;
    return {Origin::GPU, ErrType::OutOfResources, "Could not allocate GPU memory"};
  } else {
#ifdef USE_VS_MALLOC_POOLS
    deviceStats.addPtr(name, *buf, size);
#endif
    return Status::OK();
  }
}

Status freeVS(void* buf) {
#ifdef USE_VS_MALLOC_POOLS
  deviceStats.deletePtr(buf);
#endif
  if (buf) {
    return CUDA_ERROR(cudaFree(buf));
  } else {
    return Status::OK();
  }
}

Status __mallocHostVS(void** buf, size_t size, const char* name, unsigned flags, FILELINE_ARGS) {
  if (size == 0) {
    *buf = nullptr;
    return {Origin::GPU, ErrType::ImplementationError, "Cannot allocate 0-size buffer"};
  }

  if (!CUDA_ERROR(cudaHostAlloc(buf, size, flags)).ok()) {
    Logger::get(VideoStitch::Logger::Error) << "Could not allocate " << size << " bytes of pinned CPU memory.";
    PRINT_FILELINE
    Logger::get(VideoStitch::Logger::Error) << std::endl;
    return {Origin::GPU, ErrType::OutOfResources, "Could not allocate pinned memory"};
  } else {
#ifdef USE_VS_MALLOC_POOLS
    hostStats.addPtr(name, *buf, size);
#endif
    return Status::OK();
  }
}

Status freeHostVS(void* buf) {
#ifdef USE_VS_MALLOC_POOLS
  hostStats.deletePtr(buf);
#endif
  if (buf) {
    return CUDA_ERROR(cudaFreeHost(buf));
  } else {
    return Status::OK();
  }
}

Status __mallocArrayVS(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height,
                       unsigned int flags, const char* /*name*/, FILELINE_ARGS) {
  if (!CUDA_ERROR(cudaMallocArray(array, desc, width, height, flags)).ok()) {
    Logger::get(VideoStitch::Logger::Error)
        << "Could not allocate CUDA array of size " << width << " x " << height << ".";
    PRINT_FILELINE
    Logger::get(VideoStitch::Logger::Error) << std::endl;
    return {Origin::GPU, ErrType::OutOfResources, "Could not allocate CUDA array"};
  } else {
    return Status::OK();
  }
}

Status freeArrayVS(struct cudaArray* array) {
  if (array) {
    return CUDA_ERROR(cudaFreeArray(array));
  } else {
    return Status::OK();
  }
}

Status __mallocPrint(void** p, size_t size, const char* file, const int line) {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug)
      << "Alloc " << size << " CUDA bytes (" << size / (1024 * 1024) << " MB) at " << file << ":" << line << std::endl;
  return CUDA_ERROR(cudaMalloc(p, size));
}
}  // namespace Cuda
}  // namespace VideoStitch
