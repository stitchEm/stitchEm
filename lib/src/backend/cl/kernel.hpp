// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "context.hpp"

#include "cl_error.hpp"
#include "deviceBuffer.hpp"
#include "deviceStream.hpp"
#include "surface.hpp"

#include <gpu/buffer.hpp>

#include <algorithm>

inline VideoStitch::Status _setKernelParameters(cl_kernel, int) {
  return VideoStitch::Status::OK();
}  // do nothing, terminating function

template <typename T, typename... Args>
inline VideoStitch::Status _setKernelParameters(cl_kernel kernel, int i,
                                                const VideoStitch::GPU::Buffer<T>& firstParameter,
                                                const Args&... restOfParameters) {
  const cl_mem rawPtr = firstParameter.get();
  return _setKernelParameters(kernel, i, rawPtr, restOfParameters...);
}

template <typename... Args>
inline VideoStitch::Status _setKernelParameters(cl_kernel kernel, int i,
                                                const VideoStitch::GPU::DeviceSurface& firstParameter,
                                                const Args&... restOfParameters) {
  const cl_mem rawPtr = firstParameter;
  return _setKernelParameters(kernel, i, rawPtr, restOfParameters...);
}

template <typename T, typename... Args>
inline VideoStitch::Status _setKernelParameters(cl_kernel kernel, int i, const T& firstParameter,
                                                const Args&... restOfParameters) {
  static_assert(!std::is_same<T, VideoStitch::GPU::Stream>::value,
                "A GPU::Stream should never be passed as an OpenCL kernel argument");
  PROPAGATE_CL_ERR(clSetKernelArg(kernel, i, sizeof(T), &firstParameter));
  return _setKernelParameters(kernel, i + 1, restOfParameters...);
}

template <typename... Args>
inline VideoStitch::Status setKernelParameters(cl_kernel kernel, const Args&... args) {
  return _setKernelParameters(kernel, 0, args...);  // first number of parameter is 0
}

namespace VideoStitch {
namespace GPU {

template <unsigned WorkDimension>
class KernelExecution {
 public:
  template <typename... Args>
  Status enqueueWithKernelArgs(const Args&... args) {
    // lazy status evaluation, so the caller has to do less checking
    FAIL_RETURN(potKernel.getStatus());

    if (global[0] == 0) {
      return {Origin::GPU, ErrType::ImplementationError, "Trying to run kernel with work dimension x of 0"};
    }

    cl_kernel kernel = potKernel.getKernel();

    auto enqueueFunction = [&]() -> Status {
      FAIL_RETURN(setKernelParameters(kernel, args...));

      bool letDriverChooseLocalWorkSize = std::none_of(local.begin(), local.end(), [](size_t val) { return val > 0; });

      if (letDriverChooseLocalWorkSize) {
        return CL_ERROR(clEnqueueNDRangeKernel(stream.get(), kernel, WorkDimension, nullptr, global.data(), nullptr, 0,
                                               nullptr, nullptr));
      } else {
        return CL_ERROR(clEnqueueNDRangeKernel(stream.get(), kernel, WorkDimension, nullptr, global.data(),
                                               local.data(), 0, nullptr, nullptr));
      }
    };

    potKernel.lock();
    // OpenCL API:
    // "The behavior of the cl_kernel object is undefined if clSetKernelArg is called
    //  from multiple host threads on the same cl_kernel object at the same time."
    Status enqueueStatus = enqueueFunction();
    potKernel.unlock();

    return enqueueStatus;
  }

 private:
  friend class Kernel;

  KernelExecution(CLKernel& kernel, GPU::Stream stream, std::array<size_t, WorkDimension> global,
                  std::array<size_t, WorkDimension> local)
      : potKernel(kernel), stream(stream), local(local), global(global) {}

  CLKernel& potKernel;
  GPU::Stream stream;
  std::array<size_t, WorkDimension> local;
  std::array<size_t, WorkDimension> global;
};

class Kernel {
 public:
  // Request a kernel from the OpenCL context.
  // Potentially invalid: late checking usually at .enqueueWithArgs()
  // Use getInitStatus() to check for success immediately
  static Kernel get(std::string programName, std::string kernelName);

  // Does the OpenCL program exist, was the kernel found in the program
  // Did it compile succesfully?
  Status getInitStatus() const { return kernel.getStatus(); }

  // Prepare a 1D kernel to run on stream, with global work group size totalSize
  // Let the compiler choose the local work group size
  KernelExecution<1> setup1D(GPU::Stream stream, unsigned totalSize) const;

  // Prepare a 1D kernel to run on stream, with global work group size totalSize
  // Enforce a local work group size
  KernelExecution<1> setup1D(GPU::Stream stream, unsigned totalSize, unsigned blockSize) const;

  // Prepare a 2D kernel to run on stream, with global work group size totalWidth/Height
  // Let the compiler choose the local work group size
  KernelExecution<2> setup2D(GPU::Stream stream, unsigned totalWidth, unsigned totalHeight) const;

  // Prepare a 2D kernel to run on stream, with global work group size totalWidth/Height
  // Enforce a local work group size
  KernelExecution<2> setup2D(GPU::Stream stream, unsigned totalWidth, unsigned totalHeight, unsigned blockSizeX,
                             unsigned blockSizeY) const;

  // Prepare a 2D kernel to run on stream, with global work group size totalWidth/Height
  // Local work group size identical in both dimensions
  KernelExecution<2> setup2D(GPU::Stream stream, unsigned totalWidth, unsigned totalHeight, unsigned blockSize) const {
    return setup2D(stream, totalWidth, totalHeight, blockSize, blockSize);
  }

 private:
  explicit Kernel(CLKernel& kernel) : kernel(kernel) {}
  CLKernel& kernel;
};

}  // namespace GPU
}  // namespace VideoStitch
