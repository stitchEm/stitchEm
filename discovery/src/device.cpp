// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "device.hpp"

#include "deviceHelper.hpp"

#include <algorithm>
#include <cassert>
#include <ctype.h>
#include <set>
#include <sstream>
#include <string.h>

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#endif

#ifdef _MSC_VER
#include <exception>
#include <windows.h>
#include <lmerr.h>
#include <PathCch.h>

#define DELAYIMP_INSECURE_WRITABLE_HOOKS
#include "delayimp.h"
#include "winerror.h"

// define delay hook failure load function
FARPROC WINAPI failDelayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
  std::string errorstring = "Initialization error, ";

  if (dliNotify == dliFailLoadLib) {
    errorstring += "unable to load library '" + std::string(pdli->szDll) + "'";
  } else if (dliNotify == dliFailGetProc) {
    errorstring +=
        "unable to find procedure '" + std::string(pdli->dlp.szProcName) + "' in '" + std::string(pdli->szDll) + "'";
  } else {
    errorstring += "unknown error";
  }
  throw std::exception(errorstring.c_str());
}
#endif  // _MSC_VER

namespace VideoStitch {
namespace Discovery {

#ifdef CUDA_FOUND
static const int MIN_CUDA_SUPPORTED_COMPUTE_CAPABILITY[2] = {2, 0};
#endif

#ifdef OPENCL_FOUND
static const int NUM_PLATFORMS_QUERIED = 8;
static const int NUM_DEVICES_QUERIED = 16;
#endif

int getNumberOfDevices() {
  const auto& info = DevicesInfo::getInstance();
  return info.getNumberOfDevices();
}

int getNumberOfOpenCLDevices() {
  const auto& info = DevicesInfo::getInstance();
  return info.getNumberOfOpenCLDevices();
}
int getNumberOfCudaDevices() {
  const auto& info = DevicesInfo::getInstance();
  return info.getNumberOfCudaDevices();
}
bool getDeviceProperties(unsigned vsDeviceIndex, struct DeviceProperties& prop) {
  const auto& info = DevicesInfo::getInstance();
  return info.getDeviceProperties(vsDeviceIndex, prop);
}

bool getCudaDeviceProperties(unsigned vsDeviceIndex, struct DeviceProperties& prop) {
  const auto& info = DevicesInfo::getInstance();
  return info.getCudaDeviceProperties(vsDeviceIndex, prop);
}

#ifdef OPENCL_FOUND
bool getOpenCLDeviceProperties(unsigned vsDeviceIndex, struct OpenCLDevice& prop) {
  const auto& info = DevicesInfo::getInstance();
  return info.getOpenCLDeviceProperties(vsDeviceIndex, prop);
}
#endif

bool getBackendDeviceIndex(int vsDeviceIndex, int& backendDeviceIndex) {
  const auto& info = DevicesInfo::getInstance();
  return info.getBackendDeviceIndex(vsDeviceIndex, backendDeviceIndex);
}

bool getVSDeviceIndex(int backendDeviceIndex, int& vsDeviceIndex, Framework framework) {
  const auto& info = DevicesInfo::getInstance();
  return info.getVSDeviceIndex(backendDeviceIndex, vsDeviceIndex, framework);
}

FrameworkStatus getFrameworkStatus(Framework framework) {
  const auto& info = DevicesInfo::getInstance();
  return info.getFrameworkStatus(framework);
}

std::string getFrameworkName(const Framework& framework) {
  switch (framework) {
    case Discovery::Framework::CUDA:
      return "CUDA";
    case Discovery::Framework::OpenCL:
      return "OpenCL";
    case Discovery::Framework::Unknown:
    default:
      assert(false);
      return "Unknown";
  }
}

bool isFrameworkAvailable(const Framework& framework) {
  const auto& info = DevicesInfo::getInstance();
  return info.getFrameworkStatus(framework) == FrameworkStatus::Ok;
}

Framework getBestFramework(const Framework& preferedFramework) {
  const auto& info = DevicesInfo::getInstance();
  FrameworkStatus cudaStatus, openCLStatus;
  cudaStatus = info.getFrameworkStatus(Framework::CUDA);
  openCLStatus = info.getFrameworkStatus(Framework::OpenCL);

  if (cudaStatus < openCLStatus) {
    return Framework::CUDA;
  }

  if (openCLStatus < cudaStatus) {
    return Framework::OpenCL;
  }

  return preferedFramework;
}

const DevicesInfo& DevicesInfo::getInstance() {
  static DevicesInfo instance;
  return instance;
}

int DevicesInfo::getNumberOfDevices() const { return (int)genericDevices.size(); }

int DevicesInfo::getNumberOfOpenCLDevices() const {
#ifdef OPENCL_FOUND
  return (int)openCLDevices.size();
#else
  return 0;
#endif
}

int DevicesInfo::getNumberOfCudaDevices() const { return (int)cudaDevices.size(); }

bool DevicesInfo::getDeviceProperties(unsigned vsDeviceIndex, struct DeviceProperties& prop) const {
  if (vsDeviceIndex < genericDevices.size()) {
    prop = genericDevices[vsDeviceIndex];
    return true;
  }
  // this happens if the user removed their device
  return false;
}

bool DevicesInfo::getCudaDeviceProperties(unsigned vsDeviceIndex, DeviceProperties& dev) const {
  if (vsDeviceIndex < cudaDevices.size()) {
    dev = cudaDevices[vsDeviceIndex];
    return true;
  }
  return false;
}

#ifdef OPENCL_FOUND
bool DevicesInfo::getOpenCLDeviceProperties(unsigned vsDeviceIndex, OpenCLDevice& prop) const {
  if (vsDeviceIndex < openCLDevices.size()) {
    prop = openCLDevices[vsDeviceIndex];
    return true;
  }
  return false;
}
#endif  // OPENCL_FOUND

bool DevicesInfo::getBackendDeviceIndex(int vsDeviceIndex, int& backendDeviceIndex) const {
  if ((vsDeviceIndex < (int)genericDevices.size()) && (vsDeviceIndex >= 0)) {
    if (vsDeviceIndex < (int)cudaDevices.size()) {
      backendDeviceIndex = vsDeviceIndex;
    } else {
      backendDeviceIndex = vsDeviceIndex - (int)cudaDevices.size();
    }
    return true;
  }
  return false;
}

bool DevicesInfo::getVSDeviceIndex(int backendDeviceIndex, int& vsDeviceIndex, Framework framework) const {
  if (backendDeviceIndex >= 0) {
    switch (framework) {
      case Framework::CUDA:
        if (backendDeviceIndex < int(cudaDevices.size())) {
          vsDeviceIndex = backendDeviceIndex;
          return true;
        } else {
          return false;
        }
        break;
#ifdef OPENCL_FOUND
      case Framework::OpenCL:
        if (backendDeviceIndex < int(openCLDevices.size())) {
          vsDeviceIndex = backendDeviceIndex + (int)cudaDevices.size();
          return true;
        } else {
          return false;
        }
        break;
#endif
      default:
        break;
    }
  }
  return false;
}

FrameworkStatus DevicesInfo::getFrameworkStatus(Framework framework) const {
  switch (framework) {
    case Framework::CUDA:
      return cudaStatus;
    case Framework::OpenCL:
      return openCLStatus;
    case Framework::Unknown:
      assert(false);
      return FrameworkStatus::NoCompatibleDevice;
  }
  assert(false);
  return cudaStatus;
}

DevicesInfo::DevicesInfo() {
#ifdef _MSC_VER
  PfnDliHook currentFailHook = __pfnDliFailureHook2;
  __pfnDliFailureHook2 = failDelayHook;
  try {
    collectOpenCLDeviceInfo();
  } catch (std::exception& e) {
    openCLStatus = FrameworkStatus::MissingDriver;
    std::cerr << "Failed to detect OpenCL devices: " << e.what() << std::endl;
  } catch (...) {
    openCLStatus = FrameworkStatus::MissingDriver;
    std::cerr << "Failed to detect OpenCL devices!" << std::endl;
  }
  __pfnDliFailureHook2 = currentFailHook;
#else
  collectOpenCLDeviceInfo();
#endif
  collectCudaDeviceInfo();
  collectGenericDeviceInfo();
}

#ifdef OPENCL_FOUND
bool fillDeviceProperties(DeviceProperties& prop, const cl_device_id device_id) {
  cl_device_type device_type;
  cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
  if (err != CL_SUCCESS) {
    return false;
  }
  prop.compatible = true;

  // just a GPU
  if (device_type == CL_DEVICE_TYPE_GPU) {
    prop.type = DeviceType::GPU;
    // CPU or GPU/CPU?
  } else if (device_type & CL_DEVICE_TYPE_CPU) {
    prop.type = DeviceType::CPU;
  } else {
    return false;
  }

#if (defined(__APPLE__) || defined(_MSC_VER)) && !defined(OCLGRIND)
  if (prop.type == DeviceType::CPU) {
    // Intel iGPUs report as GPU on Apple machines
    // Intel CPUs do not run correctly on Apple and Windows machines
    return false;
  }
#endif  // (defined(__APPLE__) || defined(_MSC_VER)) && !defined(OCLGRIND)

  err = clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(prop.addressBits), &prop.addressBits, NULL);
  if (err != CL_SUCCESS) {
    return false;
  }

  err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(prop.globalMemSize), &prop.globalMemSize, NULL);
  if (err != CL_SUCCESS) {
    std::cout << "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed with error code " << err << std::endl;
#ifndef __ANDROID__
    return false;
#endif
  }

  char device_vendor[sizeof(prop.name)];
  size_t vendor_len;
  err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, &vendor_len);
  if (err != CL_SUCCESS) {
    return false;
  }
  std::string deviceVendor(device_vendor);
  prop.vendor = detectDeviceVendor(deviceVendor);
  if (prop.vendor == Vendor::NVIDIA) {
    // OpenCL kernels currently don't compile on Nvidia devices
    // Prefer to use CUDA backend there, don't count them as OpenCL device
    return false;
  }

  err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(prop.name), prop.name, nullptr);
  if (err != CL_SUCCESS) {
    return false;
  }

  improveDeviceName(prop);

  char openCLVersion[128];
  size_t versionSize;
  err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(openCLVersion), openCLVersion, &versionSize);
  // openCLVersion array has the following format: OpenCL<space><major_version.minor_version><space><vendor-specific
  // information>
  // check that OpenCL implementation  probably respects the format
  if ((versionSize > 9) && (openCLVersion[8] == '.')) {
    const unsigned majorVersion = atoi(&openCLVersion[7]);
    const unsigned minorVersion = atoi(&openCLVersion[9]);
    if (err != CL_SUCCESS || ((majorVersion < 2) && (minorVersion < 2))) {
      // do not support versions < 1.2
      return false;
    }
  }
  err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(prop.driverVersion), prop.driverVersion, nullptr);
  if (err != CL_SUCCESS) {
    return false;
  }
  cl_bool supportsImages;
  err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(supportsImages), &supportsImages, nullptr);
  if (err != CL_SUCCESS || !supportsImages) {
    return false;
  }
  char extension[1024];
  err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(extension), extension, nullptr);
#if !defined(OCLGRIND)
#if defined(__APPLE__)
  if (err != CL_SUCCESS || (std::string(extension).find("cl_APPLE_gl_sharing") == std::string::npos)) {
    return false;
  }
#else   // __APPLE__
  if (err != CL_SUCCESS || (std::string(extension).find("cl_khr_gl_sharing") == std::string::npos)) {
    return false;
  }
#endif  // __APPLE__
#endif  // OCLGRIND
  prop.supportedFramework = Framework::OpenCL;
  return true;
}
#endif  // OPENCL_FOUND

void DevicesInfo::collectOpenCLDeviceInfo() {
  openCLStatus = FrameworkStatus::NoCompatibleDevice;
#ifdef OPENCL_FOUND
  cl_platform_id platform_ids[NUM_PLATFORMS_QUERIED] = {NULL};

  cl_uint num_platforms;  // number of platforms available
  cl_uint num_devices;    // number of devices on selected platform

  clGetPlatformIDs(NUM_PLATFORMS_QUERIED, platform_ids, &num_platforms);

  cl_int err;
  std::set<cl_platform_id> platformsFound;
  for (cl_uint platform_index = 0; platform_index < num_platforms; platform_index++) {
    cl_platform_id platform_id = platform_ids[platform_index];
    cl_device_id device_ids[NUM_DEVICES_QUERIED] = {NULL};
    if (!platformsFound.insert(platform_id).second) {
      // the same platform may be listed several times with clGetPlatformIDs.
      continue;
    }

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, NUM_DEVICES_QUERIED, device_ids, &num_devices);
    if (err != CL_SUCCESS) {
      continue;
    }

    for (cl_uint indexDevice = 0; indexDevice < num_devices; ++indexDevice) {
      DeviceProperties prop;
      cl_device_id device_id = device_ids[indexDevice];
      if (!fillDeviceProperties(prop, device_id)) {
        continue;
      }
      size_t max_image_height;
      err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_image_height), &max_image_height,
                            nullptr);
      if (err != CL_SUCCESS) {
        continue;
      }
      openCLDevices.push_back({prop, max_image_height, platform_id, device_id});
    }
  }
  if (openCLDevices.size() != 0) {
    openCLStatus = FrameworkStatus::Ok;
  }

#if !defined(OCLGRIND)
  // Default GPU is the device at index 0, and is used for setup and default GPU actions
  // Let's use a device that is confident in its image processing capabilities
  // (CL_DEVICE_MAX_COMPUTE_UNITS would be even less useful to compare devices of different vendors)
  std::sort(openCLDevices.begin(), openCLDevices.end(), [](OpenCLDevice const& t1, OpenCLDevice const& t2) {
    if (t1.gpuProperties.type != t2.gpuProperties.type) {
      return t1.gpuProperties.type < t2.gpuProperties.type;
    }
    if (t1.gpuProperties.vendor != t2.gpuProperties.vendor) {
      return t1.gpuProperties.vendor < t2.gpuProperties.vendor;
    }
    return t1.max_image_height > t2.max_image_height;
  });
#endif  // !defined(OCLGRIND)

#endif  // OPENCL_FOUND
}

#ifdef CUDA_FOUND
static inline void resetCudaError() { cudaGetLastError(); }
#endif  // OPENCL_FOUND

void DevicesInfo::collectCudaDeviceInfo() {
  cudaStatus = FrameworkStatus::NoCompatibleDevice;
#ifdef CUDA_FOUND
  int driverVersion;
  auto result = cudaDriverGetVersion(&driverVersion);
  resetCudaError();
  if (result != cudaSuccess) {
    std::cout << "cudaDriverGetVersion failed with error code " << result << std::endl;
    cudaStatus = FrameworkStatus::MissingDriver;
    return;
  }
  int frameworkVersion;
  result = cudaRuntimeGetVersion(&frameworkVersion);
  resetCudaError();
  if (result != cudaSuccess && result != cudaErrorInsufficientDriver) {
    std::cout << "cudaRuntimeGetVersion failed with error code " << result << std::endl;
    cudaStatus = FrameworkStatus::GenericError;
    return;
  }
  if (result == cudaErrorInsufficientDriver || driverVersion < frameworkVersion) {
    std::cout << "cuda outdated driver" << std::endl;
    cudaStatus = FrameworkStatus::OutdatedDriver;
    return;
  }
  int numDevices;
  cudaError_t err = cudaGetDeviceCount(&numDevices);
  resetCudaError();
  if (err != cudaSuccess) {
    std::cout << "cudaGetDeviceCount failed with error code " << err << std::endl;
    switch (err) {
      case cudaErrorNoDevice:
        cudaStatus = FrameworkStatus::NoCompatibleDevice;
        return;
      case cudaErrorInsufficientDriver:
        cudaStatus = FrameworkStatus::OutdatedDriver;
        return;
      default:
        cudaStatus = FrameworkStatus::GenericError;
        return;
    }
  }
  char cudaDriverVersion[sizeof DeviceProperties().driverVersion];
#ifdef _MSC_VER
  int n = sprintf_s(cudaDriverVersion, "%d", driverVersion);
#else
  int n = sprintf(cudaDriverVersion, "%d", driverVersion);
#endif
  assert(n < (int)sizeof cudaDriverVersion);
  if (n < 0) {
    return;
  }
  struct cudaDeviceProp cudaProp;
  for (int i = 0; i < numDevices; i++) {
    DeviceProperties deviceProp;
    deviceProp.addressBits = 64;
    deviceProp.type = DeviceType::GPU;
    deviceProp.vendor = Vendor::NVIDIA;
    deviceProp.supportedFramework = Framework::CUDA;
    memcpy(deviceProp.driverVersion, cudaDriverVersion, sizeof cudaDriverVersion);
    if (cudaGetDeviceProperties(&cudaProp, i) != cudaSuccess) {
      resetCudaError();
      const char unknownDevice[] = "Unknown Device";
      static_assert(sizeof unknownDevice < sizeof deviceProp.name,
                    "DeviceProperties::name should be long enough to fit unknown device string");
      memcpy(deviceProp.name, unknownDevice, sizeof unknownDevice);
      deviceProp.globalMemSize = 0;
      deviceProp.compatible = false;
    } else {
      static_assert(sizeof deviceProp.name >= sizeof cudaProp.name,
                    "DeviceProperties name should be long enough to fit CUDA name");
      memcpy(deviceProp.name, cudaProp.name, sizeof cudaProp.name);
      improveDeviceName(deviceProp);
      deviceProp.globalMemSize = cudaProp.totalGlobalMem;
      deviceProp.compatible = (cudaProp.major >= MIN_CUDA_SUPPORTED_COMPUTE_CAPABILITY[0] &&
                               cudaProp.minor >= MIN_CUDA_SUPPORTED_COMPUTE_CAPABILITY[1]);
    }
    if (deviceProp.compatible) {
      cudaStatus = FrameworkStatus::Ok;
    }
    cudaDevices.push_back(deviceProp);
  }
#endif  // CUDA_FOUND
}

void DevicesInfo::collectGenericDeviceInfo() {
  for (std::vector<DeviceProperties>::iterator it = cudaDevices.begin(); it != cudaDevices.end(); ++it) {
    genericDevices.push_back(*it);
  }
#ifdef OPENCL_FOUND
  for (std::vector<OpenCLDevice>::iterator it = openCLDevices.begin(); it != openCLDevices.end(); ++it) {
    genericDevices.push_back(it->gpuProperties);
  }
#endif
}

}  // namespace Discovery
}  // namespace VideoStitch
