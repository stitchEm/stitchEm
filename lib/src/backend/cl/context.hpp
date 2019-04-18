// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cl_error.hpp"
#include "opencl.h"

#include "common/thread.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/algorithm.hpp"

#include <cstdlib>
#include <atomic>
#include <vector>
#include <unordered_map>

#include <libgpudiscovery/openCLDeviceInfo.hpp>

#include <future>

namespace VideoStitch {
namespace GPU {

class OpenCLDeviceInfo {
 public:
  PotentialValue<int> getNumberOfOpenCLDevices() const;
  PotentialValue<cl_platform_id> getPlatform(int backendDeviceIndex) const;
  PotentialValue<Discovery::DeviceProperties> getDeviceProperties(int backendDeviceIndex) const;
  PotentialValue<cl_device_id> getOpenCLDeviceID(int backendDeviceIndex) const;
  static const OpenCLDeviceInfo &getInstance();

 private:
  OpenCLDeviceInfo();
  Status collectDeviceInfo();

  std::vector<Discovery::OpenCLDevice> devices;
  Status deviceQueryStatus;
};

class CLKernel {
 public:
  explicit CLKernel(PotentialValue<cl_kernel> kernel) : kernel(kernel), mutex() {}

  Status getStatus() { return kernel.status(); }

  cl_kernel getKernel() {
    assert(kernel.status().ok());
    return kernel.value();
  }

  void lock() { mutex.lock(); }

  void unlock() { mutex.unlock(); }

 private:
  PotentialValue<cl_kernel> kernel;
  std::mutex mutex;
};

class OpenCLContext;

class CLProgram {
 public:
  CLProgram(std::string programName, const char *program_binary_32, size_t binary_32_length,
            const char *program_binary_64, size_t binary_64_length,
#ifdef ALTERNATIVE_OPENCL_SPIR
            const char *program_binary_32_alternative, size_t binary_32_length_alternative,
            const char *program_binary_64_alternative, size_t binary_64_length_alternative,
#endif
            const bool likelyUsed)
      : mutex(),
        programName(programName),
        binary32(program_binary_32),
        binaryLength32(binary_32_length),
        binary64(program_binary_64),
        binaryLength64(binary_64_length),
#ifdef ALTERNATIVE_OPENCL_SPIR
        alternativeBinary32(program_binary_32_alternative),
        alternativeBinaryLength32(binary_32_length_alternative),
        alternativeBinary64(program_binary_64_alternative),
        alternativeBinaryLength64(binary_64_length_alternative),
#endif
        likelyUsed(likelyUsed),
        compiled(false),
        cached(false),
        searched(false),
        program(nullptr) {
  }

  CLKernel &getKernel(std::string kernelName) {
    std::lock_guard<std::mutex> lock(mutex);

    auto found = kernels.find(kernelName);
    if (found == kernels.end()) {
      if (!compilationStatus.ok()) {
        kernels.emplace(kernelName, std::make_unique<CLKernel>(compilationStatus));
      } else {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
        if (err != CL_SUCCESS) {
          Status kernelStatus{Origin::GPU, ErrType::UnsupportedAction,
                              "Cannot create kernel " + kernelName + " in program " + programName, CL_ERROR(err)};
          kernels.emplace(kernelName, std::make_unique<CLKernel>(kernelStatus));
        }
        kernels.emplace(kernelName, std::make_unique<CLKernel>(kernel));
      }
    }
    return *kernels.at(kernelName).get();
  }

  bool isCompiled() const { return compiled; }

  bool isLikelyUsed() const { return likelyUsed; }

  bool isCached() const { return cached; }

  std::string getName() const { return programName; }

  PotentialValue<size_t> getSourceLength(const Discovery::DeviceProperties &prop);
  PotentialValue<size_t> getAlternativeSourceLength(const Discovery::DeviceProperties &prop);
  Status compile(OpenCLContext &context);

  std::vector<std::string> queryKernelNames();
  PotentialValue<size_t> sizeToCompile(OpenCLContext &context);

 private:
  std::mutex mutex;
  const std::string programName;
  const char *binary32;
  size_t binaryLength32;
  const char *binary64;
  size_t binaryLength64;
#ifdef ALTERNATIVE_OPENCL_SPIR
  const char *alternativeBinary32;
  size_t alternativeBinaryLength32;
  const char *alternativeBinary64;
  size_t alternativeBinaryLength64;
#endif
  const bool likelyUsed;
  std::atomic<bool> compiled;
  bool cached;
  bool searched;
  Status compilationStatus;
  cl_program program;
  std::unordered_map<std::string, std::unique_ptr<CLKernel>> kernels;

  void searchProgram(OpenCLContext &context, Discovery::DeviceProperties &prop, const cl_device_id &device_id,
                     const char *source, size_t sourceLength);
  PotentialValue<const char *> getSource(const Discovery::DeviceProperties &prop);
  PotentialValue<const char *> getAlternativeSource(const Discovery::DeviceProperties &prop);
};

PotentialValue<OpenCLContext> getContext();
Status destroyOpenCLContext();

class OpenCLContext {
 public:
  static PotentialValue<OpenCLContext> getInstance();
  explicit OpenCLContext(int backendDeviceID);

  // This default constructor has only been created for PotentialValue constructor.
  // It should be called only if something goes wrong !
  OpenCLContext();

  // Retrieve a compiled OpenCL kernel by name.
  // If the kernel has not been used yet, this call might block
  // while the kernel is compiled for the first time.
  CLKernel &getKernel(std::string program_name, std::string kernel_name);

  operator cl_context() const { return context; }

  cl_device_id deviceID() const { return device_id; }

  int get_backendDeviceId() const { return backendDeviceID; }

  // Force the compilation of all referenced OpenCL programs
  Status compileAllPrograms(Util::Algorithm::ProgressReporter *pReporter, const bool aheadOfTime);

  Status getStatus() const { return contextStatus; }

 private:
  void operator=(OpenCLContext const &) = delete;

  PotentialValue<size_t> computeTotSizeToCompile(const bool aheadOfTime);

  std::unordered_map<std::string, cl_program> programs;
  Status contextStatus;

  cl_context context;
  cl_device_id device_id;
  int backendDeviceID;
};

class ProgramRegistry {
 public:
  typedef std::unordered_map<std::string, CLProgram> ProgramMap;

  ProgramRegistry(std::string program_name, const char *program_binary_32, size_t binary_32_length,
                  const char *program_binary_64, size_t binary_64_length,
#ifdef ALTERNATIVE_OPENCL_SPIR
                  const char *program_binary_32_alternative, size_t binary_32_length_alternative,
                  const char *program_binary_64_alternative, size_t binary_64_length_alternative,
#endif
                  const bool likelyUsed) {
    // make sure the same name is not registered twice
    assert(programRegister().find(program_name) == programRegister().end());
    programRegister().emplace(
        std::piecewise_construct, std::forward_as_tuple(program_name),
        std::forward_as_tuple(program_name, program_binary_32, binary_32_length, program_binary_64, binary_64_length,
#ifdef ALTERNATIVE_OPENCL_SPIR
                              program_binary_32_alternative, binary_32_length_alternative,
                              program_binary_64_alternative, binary_64_length_alternative,
#endif
                              likelyUsed));
  }

  static CLProgram &get(std::string program_name) {
    // if you hit this assertion, you are trying to access a program that has not been registered properly
    if (programRegister().find(program_name) == programRegister().end()) {
      Logger::error("OpenCL") << "Trying to use OpenCL program " << program_name << " which was not registered!"
                              << std::endl;
      std::abort();
    }
    return programRegister().at(program_name);
  }

  static const ProgramMap &getAllPrograms() { return programRegister(); }

 private:
  friend class OpenCLContext;

  static ProgramMap &programRegister() {
    static ProgramMap pmap;
    return pmap;
  }
};

#define STRINGIFY(x) #x
#define KERNEL_STR(k) STRINGIFY(k)

#ifdef DISABLE_OPENCL_SPIR
// Source is the same for 32 and 64 bit, just store it twice
#define REGISTER_OPENCL_PROGRAM(CL_PROGRAM_NAME, likelyUsed)                                \
  VideoStitch::GPU::ProgramRegistry register_##CL_PROGRAM_NAME(                             \
      #CL_PROGRAM_NAME, (const char *)&CL_PROGRAM_NAME##_pre[0], CL_PROGRAM_NAME##_pre_len, \
      (const char *)&CL_PROGRAM_NAME##_pre[0], CL_PROGRAM_NAME##_pre_len, likelyUsed);

#else  // DISABLE_OPENCL_SPIR
#ifdef ALTERNATIVE_OPENCL_SPIR
#define REGISTER_OPENCL_PROGRAM(CL_PROGRAM_NAME, likelyUsed)                                            \
  VideoStitch::GPU::ProgramRegistry register_##CL_PROGRAM_NAME(                                         \
      #CL_PROGRAM_NAME, (const char *)&CL_PROGRAM_NAME##_spir32[0], CL_PROGRAM_NAME##_spir32_len,       \
      (const char *)&CL_PROGRAM_NAME##_spir64[0], CL_PROGRAM_NAME##_spir64_len,                         \
      (const char *)&CL_PROGRAM_NAME##_alternative_spir32[0], CL_PROGRAM_NAME##_alternative_spir32_len, \
      (const char *)&CL_PROGRAM_NAME##_alternative_spir64[0], CL_PROGRAM_NAME##_alternative_spir64_len, likelyUsed);
#else  // ALTERNATIVE_OPENCL_SPIR
#define REGISTER_OPENCL_PROGRAM(CL_PROGRAM_NAME, likelyUsed)                                      \
  VideoStitch::GPU::ProgramRegistry register_##CL_PROGRAM_NAME(                                   \
      #CL_PROGRAM_NAME, (const char *)&CL_PROGRAM_NAME##_spir32[0], CL_PROGRAM_NAME##_spir32_len, \
      (const char *)&CL_PROGRAM_NAME##_spir64[0], CL_PROGRAM_NAME##_spir64_len, likelyUsed);

#endif  // ALTERNATIVE_OPENCL_SPIR
#endif  // DISABLE_OPENCL_SPIR

#define INDIRECT_REGISTER_OPENCL_PROGRAM(A, likelyUsed) REGISTER_OPENCL_PROGRAM(A, likelyUsed)

#define PROGRAM(k) STRINGIFY(k)

}  // namespace GPU
}  // namespace VideoStitch
