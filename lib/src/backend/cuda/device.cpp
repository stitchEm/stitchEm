// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/gpu_device.hpp"

#include "gpu/buffer.hpp"

#include "cuda/memory.hpp"
#include "cuda/error.hpp"
#include <libgpudiscovery/cudaDeviceInfo.hpp>

#include <cuda_runtime.h>
#if !defined(__APPLE__) && !defined(TEGRA_DEMO)
#include <nvml.h>
#endif

#include <algorithm>
#include <functional>
#include <atomic>

#if defined(linux)
#include <sys/types.h>
#include <unistd.h>
#elif _MSC_VER
#include <Windows.h>
#endif

namespace VideoStitch {

#if !defined(__APPLE__) && !defined(TEGRA_DEMO)
namespace {

void init();
void done();

struct NVML {
  NVML() { init(); }
  ~NVML() { done(); }

  unsigned int device_count = 0;
};

NVML nvml;
#ifdef linux
pid_t pid;
#elif _MSC_VER
DWORD pid;
#endif

void init() {
  nvmlReturn_t result = nvmlInit();
  if (NVML_SUCCESS != result) {
    printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
    return;
  }
  result = nvmlDeviceGetCount(&nvml.device_count);
  if (NVML_SUCCESS != result) {
    printf("Failed to query device count: %s\n", nvmlErrorString(result));
    return;
  }
  printf("Found %d device%s\n\n", (int)nvml.device_count, nvml.device_count != 1 ? "s" : "");

#ifdef linux
  pid = getpid();
#elif _MSC_VER
  pid = GetCurrentProcessId();
#endif
}

void done() {
  nvmlReturn_t result = nvmlInit();
  if (NVML_SUCCESS != result) {
    printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
  }
}
}  // namespace
#endif  // __APPLE__

namespace GPU {

std::atomic<int> defaultDevice(-1);

Status checkDefaultBackendDeviceInitialization() { return CUDA_ERROR(cudaPeekAtLastError()); }

Status setDefaultBackendDeviceVS(int vsDevice) {
  int device;
  if (!Discovery::getBackendDeviceIndex(vsDevice, device)) {
    return {Origin::GPU, ErrType::ImplementationError, "[CUDA] trying to set a device that does not exist"};
  }
  return setDefaultBackendDevice(device);
}

Status getDefaultBackendDeviceContext(void* context) { return CUDA_ERROR(cuCtxGetCurrent((CUcontext*)context)); }

Status useDefaultBackendDevice() {
  if (defaultDevice == -1) {
    return {Origin::GPU, ErrType::ImplementationError, "[CUDA] default device is not set"};
  }
  return CUDA_ERROR(cudaSetDevice(defaultDevice));
}

Status setDefaultBackendDevice(int device) {
  if (defaultDevice != -1 && device != defaultDevice) {
    return {Origin::GPU, ErrType::ImplementationError, "[CUDA] changing the default device is not permitted"};
  }
  defaultDevice = device;
  return useDefaultBackendDevice();
}

Status getDefaultBackendDevice(int* device) {
  if (defaultDevice == -1) {
    return {Origin::GPU, ErrType::ImplementationError, "[CUDA] default device is not set"};
  }
  *device = defaultDevice;
  return Status::OK();
}

Discovery::Framework getFramework() { return Discovery::Framework::CUDA; }

PotentialValue<size_t> getMemoryUsage() {
  // add buffers, cached buffers and allocated size on device
  size_t used_memory = 0;
  used_memory += getBufferPoolCurrentSize();
  // used_memory += getCachedBufferPoolCurrentSize();
  used_memory += Cuda::getDevicePoolCurrentSize();
  return PotentialValue<size_t>(used_memory);
}

PotentialValue<std::vector<size_t> > getMemoryUsageByDevices() {
  std::vector<size_t> used_memory;

#if !defined(__APPLE__) && !defined(TEGRA_DEMO)
  for (unsigned int i = 0; i < nvml.device_count; i++) {
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(i, &device);

#ifdef linux
    unsigned int infoCount = 0;
    nvmlDeviceGetComputeRunningProcesses(device, &infoCount, nullptr);
    nvmlProcessInfo_t* infos = new nvmlProcessInfo_t[infoCount];
    nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos);

    for (unsigned int j = 0; j < infoCount; j++) {
      if ((unsigned int)pid == infos[j].pid) {
        used_memory.push_back(infos[j].usedGpuMemory);
      }
    }
    delete[] infos;
#elif _MSC_VER
    nvmlMemory_t memory;
    nvmlDeviceGetMemoryInfo(device, &memory);
    used_memory.push_back(memory.used);
#endif
  }
#else
  // add buffers, cached buffers and allocated size on device
  // initialize with first buffers
  used_memory = getBufferPoolCurrentSizeByDevices();
  // accumulate other buffers
  // std::transform(used_memory.begin(), used_memory.end(), getCachedBufferPoolCurrentSizeByDevices().begin(),
  // used_memory.begin(), std::plus<size_t>());
  std::transform(used_memory.begin(), used_memory.end(), Cuda::getDevicePoolCurrentSizeByDevices().begin(),
                 used_memory.begin(), std::plus<size_t>());
#endif

  return PotentialValue<std::vector<size_t> >(used_memory);
}

}  // namespace GPU
}  // namespace VideoStitch
