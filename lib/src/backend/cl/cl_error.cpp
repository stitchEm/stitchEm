// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cl_error.hpp"

namespace VideoStitch {
namespace GPU {

Status clStatusHelper(Origin o, ErrType err, const std::string message, const char* file, const int line) {
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

Status clErrorStatus(int error, const char* file, const int line) {
  switch (error) {
    case CL_SUCCESS:
      return Status::OK();
    case CL_DEVICE_NOT_FOUND:
      return clStatusHelper(Origin::GPU, ErrType::InvalidConfiguration, "Device not found", file, line);
    case CL_DEVICE_NOT_AVAILABLE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Device not available", file, line);
    case CL_COMPILER_NOT_AVAILABLE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Compiler not available", file, line);
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return clStatusHelper(Origin::GPU, ErrType::OutOfResources,
                            "Mem object allocation failure. Reduce the project output size and close other "
                            "applications to free up GPU memory.",
                            file, line);
    case CL_OUT_OF_RESOURCES:
      return clStatusHelper(
          Origin::GPU, ErrType::OutOfResources,
          "Out of resources. Reduce the project output size and close other applications to free up GPU resources.",
          file, line);
    case CL_OUT_OF_HOST_MEMORY:
      return clStatusHelper(
          Origin::GPU, ErrType::OutOfResources,
          "Out of host memory. Reduce the project output size and close other applications to free up RAM.", file,
          line);
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "profiling info not available", file, line);
    case CL_MEM_COPY_OVERLAP:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Mem copy overlap", file, line);
    case CL_IMAGE_FORMAT_MISMATCH:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Image format mismatch", file, line);
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Image format not supported", file, line);
    case CL_BUILD_PROGRAM_FAILURE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Build program failure", file, line);
    case CL_MAP_FAILURE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Map failure", file, line);
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Misaligned sub buffer offset", file, line);
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Exec status error for events in wait list", file,
                            line);
    case CL_COMPILE_PROGRAM_FAILURE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Compile program failure", file, line);
    case CL_LINKER_NOT_AVAILABLE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Linker not available", file, line);
    case CL_LINK_PROGRAM_FAILURE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Link program failure", file, line);
    case CL_DEVICE_PARTITION_FAILED:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Device partition failed", file, line);
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Kernel arg info not available", file, line);
    case CL_INVALID_VALUE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid value", file, line);
    case CL_INVALID_DEVICE_TYPE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid device type", file, line);
    case CL_INVALID_PLATFORM:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid platform", file, line);
    case CL_INVALID_DEVICE:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid device", file, line);
    case CL_INVALID_CONTEXT:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid context", file, line);
    case CL_INVALID_QUEUE_PROPERTIES:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid queue properties", file, line);
    case CL_INVALID_COMMAND_QUEUE:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid command queue", file, line);
    case CL_INVALID_HOST_PTR:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid host ptr", file, line);
    case CL_INVALID_MEM_OBJECT:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid mem object", file, line);
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid image format descriptor", file, line);
    case CL_INVALID_IMAGE_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid image size", file, line);
    case CL_INVALID_SAMPLER:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid sampler", file, line);
    case CL_INVALID_BINARY:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid binary", file, line);
    case CL_INVALID_BUILD_OPTIONS:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid build options", file, line);
    case CL_INVALID_PROGRAM:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid program", file, line);
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid program executable", file, line);
    case CL_INVALID_KERNEL_NAME:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid kernel name", file, line);
    case CL_INVALID_KERNEL_DEFINITION:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid kernel definition", file, line);
    case CL_INVALID_KERNEL:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid kernel", file, line);
    case CL_INVALID_ARG_INDEX:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid arg index", file, line);
    case CL_INVALID_ARG_VALUE:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid arg value", file, line);
    case CL_INVALID_ARG_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid arg size", file, line);
    case CL_INVALID_KERNEL_ARGS:
      return clStatusHelper(Origin::GPU, ErrType::ImplementationError, "Invalid kernel args", file, line);
    case CL_INVALID_WORK_DIMENSION:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid work dimension", file, line);
    case CL_INVALID_WORK_GROUP_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid work group size", file, line);
    case CL_INVALID_WORK_ITEM_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid work item size", file, line);
    case CL_INVALID_GLOBAL_OFFSET:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid global offset", file, line);
    case CL_INVALID_EVENT_WAIT_LIST:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid event wait list", file, line);
    case CL_INVALID_EVENT:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid event", file, line);
    case CL_INVALID_OPERATION:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid operation", file, line);
    case CL_INVALID_GL_OBJECT:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid gl object", file, line);
    case CL_INVALID_BUFFER_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::OutOfResources, "Invalid buffer size", file, line);
    case CL_INVALID_MIP_LEVEL:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid mip level", file, line);
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid global work size", file, line);
    case CL_INVALID_PROPERTY:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid property", file, line);
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid image descriptor", file, line);
    case CL_INVALID_COMPILER_OPTIONS:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid compiler options", file, line);
    case CL_INVALID_LINKER_OPTIONS:
      return clStatusHelper(Origin::GPU, ErrType::UnsupportedAction, "Invalid linker options", file, line);
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Invalid device partition count", file, line);
#ifdef __APPLE__
    case CL_INVALID_GL_CONTEXT_APPLE:
#else
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
#endif
      return clStatusHelper(Origin::GPU, ErrType::InvalidConfiguration,
                            "Selected GPU device can not display. Connect it to the monitor.", file, line);
    default:
      return clStatusHelper(Origin::GPU, ErrType::RuntimeError, "Unknown OpenCL error code", file, line);
  }
}

Status checkErrorStatus(int errorCode, const char* file, int line) {
  if (errorCode == CL_SUCCESS) {
    return Status::OK();
  } else {
    return clErrorStatus(errorCode, file, line);
  }
}

}  // namespace GPU
}  // namespace VideoStitch
