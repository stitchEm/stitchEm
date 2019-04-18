// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "error.hpp"

#include "libvideostitch/logging.hpp"

#include <cstdlib>
#include <sstream>

namespace VideoStitch {
namespace Cuda {

Status cudaStatusHelper(Origin o, ErrType err, const std::string& message, const char* file, const int line) {
#if defined(_MSC_VER)
  (void)file;
  (void)line;
#endif
  return {o, err,
#ifndef NDEBUG
          std::string(file) + " (" + std::to_string(line) + "): " +
#endif
              message};
}

Status cudaStatus(cudaError err, const char* file, const int line) {
  // reset cuda error
  cudaGetLastError();
  switch (err) {
    case cudaSuccess:
      return Status::OK();
    case cudaErrorMissingConfiguration:
      return cudaStatusHelper(Origin::GPU, ErrType::ImplementationError, "Missing configuration", file, line);
    case cudaErrorMemoryAllocation:
      return cudaStatusHelper(
          Origin::GPU, ErrType::OutOfResources,
          "Out of resources. Reduce the project output size and close other applications to free up GPU resources.",
          file, line);
    case cudaErrorInitializationError:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Initialization error", file, line);
    case cudaErrorLaunchFailure:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch failure", file, line);
    case cudaErrorPriorLaunchFailure:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Previous launch failed", file, line);
    case cudaErrorLaunchTimeout:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch timeout", file, line);
    case cudaErrorLaunchOutOfResources:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Launch failed, unappropriate resources",
                              file, line);
    case cudaErrorInvalidDeviceFunction:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid device function", file, line);
    case cudaErrorInvalidConfiguration:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid configuration", file, line);
    case cudaErrorInvalidDevice:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid device id", file, line);
    case cudaErrorInvalidValue:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid value", file, line);
    case cudaErrorInvalidPitchValue:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid pitch value", file, line);
    case cudaErrorInvalidSymbol:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid symbol name/identifier", file, line);
    case cudaErrorMapBufferObjectFailed:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Buffer object could not be mapped", file, line);
    case cudaErrorUnmapBufferObjectFailed:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Buffer object could not be unmapped", file, line);
    case cudaErrorInvalidHostPointer:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid host pointer", file, line);
    case cudaErrorInvalidDevicePointer:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid device pointer", file, line);
    case cudaErrorInvalidTexture:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid texture", file, line);
    case cudaErrorInvalidTextureBinding:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid texture binding", file, line);
    case cudaErrorInvalidChannelDescriptor:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid channel descriptor", file, line);
    case cudaErrorInvalidMemcpyDirection:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid memcpy direction", file, line);
    case cudaErrorAddressOfConstant:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid address", file, line);
    case cudaErrorTextureFetchFailed:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Texture fetch failed", file, line);
    case cudaErrorTextureNotBound:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Texture not bound for access", file, line);
    case cudaErrorSynchronizationError:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Synchronization failed", file, line);
    case cudaErrorInvalidFilterSetting:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid filter setting", file, line);
    case cudaErrorInvalidNormSetting:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid normalized float", file, line);
    case cudaErrorMixedDeviceExecution:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Mixed device execution", file, line);
    case cudaErrorCudartUnloading:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Cuda driver unloaded", file, line);
    case cudaErrorNotYetImplemented:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Not yet implemented", file, line);
    case cudaErrorMemoryValueTooLarge:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError,
                              "Emulated device pointer exceeded the 32-bit address range", file, line);
    case cudaErrorInvalidResourceHandle:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid resource handle", file, line);
    case cudaErrorNotReady:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Not ready", file, line);
    case cudaErrorInsufficientDriver:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration,
                              "Nvidia CUDA driver is older than the CUDA runtime library", file, line);
    case cudaErrorSetOnActiveProcess:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Set on active process", file, line);
    case cudaErrorInvalidSurface:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid surface", file, line);
    case cudaErrorNoDevice:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Device not found", file, line);
    case cudaErrorECCUncorrectable:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Uncorrectable ECC error", file, line);
    case cudaErrorSharedObjectSymbolNotFound:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Failed to resolve shared object symbol", file, line);
    case cudaErrorSharedObjectInitFailed:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Shared object initilialization failed", file, line);
    case cudaErrorUnsupportedLimit:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Unsupported limit", file, line);
    case cudaErrorDuplicateVariableName:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Duplicate varaiable name", file, line);
    case cudaErrorDuplicateTextureName:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Duplicate texture name", file, line);
    case cudaErrorDuplicateSurfaceName:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Duplicate surface name", file, line);
    case cudaErrorDevicesUnavailable:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Device unavailable", file, line);
    case cudaErrorInvalidKernelImage:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid kernel image", file, line);
    case cudaErrorNoKernelImageForDevice:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "No kernel image for the device", file, line);
    case cudaErrorIncompatibleDriverContext:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Incompatible driver context", file, line);
    case cudaErrorPeerAccessAlreadyEnabled:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Peer access already enabled", file, line);
    case cudaErrorPeerAccessNotEnabled:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Peer access not enabled", file, line);
    case cudaErrorDeviceAlreadyInUse:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Device already in use", file, line);
    case cudaErrorProfilerDisabled:
    case cudaErrorProfilerNotInitialized:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Profiler not initialized", file, line);
    case cudaErrorProfilerAlreadyStarted:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Profiler already started", file, line);
    case cudaErrorProfilerAlreadyStopped:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Profiler already stopped", file, line);
    case cudaErrorAssert:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Assert error", file, line);
    case cudaErrorTooManyPeers:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Too many peers", file, line);
    case cudaErrorHostMemoryAlreadyRegistered:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Memory range already registered", file, line);
    case cudaErrorHostMemoryNotRegistered:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid host memory region", file, line);
    case cudaErrorOperatingSystem:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "OS call failed", file, line);
    case cudaErrorPeerAccessUnsupported:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "P2P access not supported", file, line);
    case cudaErrorLaunchMaxDepthExceeded:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch maximum depth exceeded", file, line);
    case cudaErrorLaunchFileScopedTex:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unsupported file-scoped textures", file, line);
    case cudaErrorLaunchFileScopedSurf:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unsupported file-scoped surfaces", file, line);
    case cudaErrorSyncDepthExceeded:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Synchronize depth exceeded", file, line);
    case cudaErrorLaunchPendingCountExceeded:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch exceeds pending count", file, line);
    case cudaErrorNotPermitted:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Operation not permitted", file, line);
    case cudaErrorNotSupported:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Operation not supported", file, line);
    case cudaErrorHardwareStackError:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Stack corruption", file, line);
    case cudaErrorIllegalInstruction:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Illegal instruction", file, line);
    case cudaErrorMisalignedAddress:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Misaligned memory address", file, line);
    case cudaErrorInvalidAddressSpace:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid memory address", file, line);
    case cudaErrorInvalidPc:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid program counter", file, line);
    case cudaErrorIllegalAddress:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Illegal address", file, line);
    case cudaErrorInvalidPtx:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid PTX", file, line);
    case cudaErrorInvalidGraphicsContext:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid graphic context", file, line);
    case cudaErrorStartupFailure:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Startup failure", file, line);
    case cudaErrorApiFailureBase:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Unhandled driver error", file, line);
    case cudaErrorUnknown:
    default:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unkown error", file, line);
  }
}

Status cudaStatus(CUresult code, const char* file, const int line) {
  // reset cuda error
  cudaGetLastError();
  switch (code) {
    case CUDA_SUCCESS:
      return Status::OK();
    case CUDA_ERROR_INVALID_VALUE:
      return cudaStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid values", file, line);
    case CUDA_ERROR_OUT_OF_MEMORY:
      return cudaStatusHelper(
          Origin::GPU, ErrType::OutOfResources,
          "Out of resources. Reduce the project output size and close other applications to free up GPU resources.",
          file, line);
    case CUDA_ERROR_NOT_INITIALIZED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver is not initialized", file, line);
    case CUDA_ERROR_DEINITIALIZED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver is shutting down", file, line);
    case CUDA_ERROR_PROFILER_DISABLED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver profiler is disabled", file, line);
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver profiler is not initialized", file, line);
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver profiler already started", file, line);
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Driver profiler already stopped", file, line);
    case CUDA_ERROR_NO_DEVICE:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Device not found", file, line);
    case CUDA_ERROR_INVALID_DEVICE:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid device", file, line);
    case CUDA_ERROR_INVALID_IMAGE:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid kernel image", file, line);
    case CUDA_ERROR_INVALID_CONTEXT:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid context", file, line);
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Supplied context is already active", file, line);
    case CUDA_ERROR_MAP_FAILED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Map failure", file, line);
    case CUDA_ERROR_UNMAP_FAILED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unmap failed", file, line);
    case CUDA_ERROR_ARRAY_IS_MAPPED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Cannot destroy a currently mapped array", file,
                              line);
    case CUDA_ERROR_ALREADY_MAPPED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Resource is already mapped", file, line);
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "No kernel image available for the device",
                              file, line);
    case CUDA_ERROR_ALREADY_ACQUIRED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Resource already acquired", file, line);
    case CUDA_ERROR_NOT_MAPPED:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Resource not mapped", file, line);
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Resource not available as an array", file, line);
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Resource not available as a pointer", file, line);
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return cudaStatusHelper(Origin::GPU, ErrType::ImplementationError, "Uncorrectable ECC error", file, line);
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Unsopported limit", file, line);
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Context already in use", file, line);
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Peer access is not supported", file, line);
    case CUDA_ERROR_INVALID_PTX:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "PTX JIT compilation failed", file, line);
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid graphic context", file, line);
    case CUDA_ERROR_INVALID_SOURCE:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid device kernel source", file, line);
    case CUDA_ERROR_FILE_NOT_FOUND:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "File not found", file, line);
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Shared object symbol not found", file, line);
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Shared object initialization failed", file, line);
    case CUDA_ERROR_OPERATING_SYSTEM:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "OS call failed", file, line);
    case CUDA_ERROR_INVALID_HANDLE:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Invalid resource handle", file, line);
    case CUDA_ERROR_NOT_FOUND:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Symbol not found", file, line);
    case CUDA_ERROR_NOT_READY:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Device not ready", file, line);
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid memory address", file, line);
      /**  FROM cuda.h
       * This indicates that a launch did not occur because it did not have
       * appropriate resources. This error usually indicates that the user has
       * attempted to pass too many arguments to the device kernel, or the
       * kernel launch specifies too many threads for the kernel's register
       * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
       * when a 32-bit int is expected) is equivalent to passing too many
       * arguments and can also result in this error.
       */
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Launch failed, out of resources", file,
                              line);
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch timeout", file, line);
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Incompatible texturing mode", file, line);
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Peer access already enabled", file, line);
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Peer access not enabled", file, line);
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Primary context already initialized", file, line);
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Primary context destroyed or not yet initialized",
                              file, line);
    case CUDA_ERROR_ASSERT:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Assert during kernel execution", file, line);
    case CUDA_ERROR_TOO_MANY_PEERS:
      return cudaStatusHelper(Origin::GPU, ErrType::OutOfResources, "Out of resources. Too may peers", file, line);
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Host memory already registered", file, line);
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      return cudaStatusHelper(Origin::GPU, ErrType::InvalidConfiguration,
                              "Pointer does not correspond to any memory region", file, line);
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
      return cudaStatusHelper(Origin::GPU, ErrType::OutOfResources, "Out of resources. Stack error", file, line);
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Illegal instruction", file, line);
    case CUDA_ERROR_MISALIGNED_ADDRESS:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Misaligned memory address", file, line);
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid address space", file, line);
    case CUDA_ERROR_INVALID_PC:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid program counter", file, line);
    case CUDA_ERROR_LAUNCH_FAILED:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Launch failed", file, line);
    case CUDA_ERROR_NOT_PERMITTED:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Operation is not permitted", file, line);
    case CUDA_ERROR_NOT_SUPPORTED:
      return cudaStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Unsupported operation", file, line);
    case CUDA_ERROR_UNKNOWN:
    default:
      return cudaStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unknown internal error", file, line);
  }
}

Status cudaCheckError(CUresult err, const char* file, const int line) {
  if (err == CUDA_SUCCESS) {
    return Status::OK();
  } else {
    return cudaStatus(err, file, line);
  }
}

Status cudaCheckError(cudaError err, const char* file, const int line) {
  if (err == cudaSuccess) {
    return Status::OK();
  } else {
    return cudaStatus(err, file, line);
  }
}

Status cudaCheckStatus(const char* file, int line) {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    return cudaStatus(err, file, line);
  }
  return Status::OK();
}

}  // namespace Cuda
}  // namespace VideoStitch
