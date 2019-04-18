// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "context.hpp"

#include "libvideostitch/logging.hpp"
#include "binaryCache.hpp"

#include <algorithm>
#include <future>
#include <set>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#include <Wingdi.h>
#elif __ANDROID__
#include <EGL/egl.h>
#elif __linux__
#include <GL/glx.h>
#undef Status
#elif __APPLE__
#include <OpenGL/OpenGL.h>
#endif

namespace VideoStitch {
namespace GPU {

static std::mutex contextMutex;
static bool destroyed = false;

PotentialValue<int> OpenCLDeviceInfo::getNumberOfOpenCLDevices() const {
  if (!deviceQueryStatus.ok()) {
    return deviceQueryStatus;
  }
  return (int)devices.size();
}

PotentialValue<cl_device_id> OpenCLDeviceInfo::getOpenCLDeviceID(int backendDeviceIndex) const {
  if (!deviceQueryStatus.ok()) {
    return deviceQueryStatus;
  }
  if (backendDeviceIndex < 0 || (size_t)backendDeviceIndex > devices.size()) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Querying OpenCL device at invalid index"};
  }
  return devices[backendDeviceIndex].device_id;
}

PotentialValue<Discovery::DeviceProperties> OpenCLDeviceInfo::getDeviceProperties(int backendDeviceIndex) const {
  if (!deviceQueryStatus.ok()) {
    return deviceQueryStatus;
  }
  if (backendDeviceIndex < 0 || (size_t)backendDeviceIndex >= devices.size()) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Querying OpenCL device at invalid index"};
  }
  return PotentialValue<Discovery::DeviceProperties>(devices[backendDeviceIndex].gpuProperties);
}

PotentialValue<cl_platform_id> OpenCLDeviceInfo::getPlatform(int backendDeviceIndex) const {
  if (!deviceQueryStatus.ok()) {
    return deviceQueryStatus;
  }
  if (backendDeviceIndex < 0 || (size_t)backendDeviceIndex > devices.size()) {
    return Status{Origin::GPU, ErrType::ImplementationError, "Querying OpenCL device at invalid index"};
  }
  return devices[backendDeviceIndex].platform_id;
}

OpenCLDeviceInfo::OpenCLDeviceInfo() { deviceQueryStatus = collectDeviceInfo(); }

Status OpenCLDeviceInfo::collectDeviceInfo() {
  int numberOfOpenCLDevices = Discovery::getNumberOfOpenCLDevices();
  if (!numberOfOpenCLDevices) {
    return {Origin::GPU, ErrType::SetupFailure,
            "No valid OpenCL devices found. Please make sure you have installed an OpenCL capable device that matches "
            "the minimum hardware requirements."};
  }
  for (int i = 0; i < numberOfOpenCLDevices; i++) {
    Discovery::OpenCLDevice device;
    if (Discovery::getOpenCLDeviceProperties(i, device)) {
      devices.push_back(device);
    }
  }
  return Status::OK();
}

const OpenCLDeviceInfo& OpenCLDeviceInfo::getInstance() {
  static OpenCLDeviceInfo instance;
  return instance;
}

Status destroyOpenCLContext() {
  const auto& potContext = getContext();
  std::lock_guard<std::mutex> lock(contextMutex);
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  destroyed = true;
  return (CL_ERROR(clReleaseContext(cl_context(ctx))));
}

PotentialValue<OpenCLContext> getContext() { return OpenCLContext::getInstance(); }

#if (_MSC_VER && _MSC_VER < 1900)
// C++11 magic statics support from Visual Studio 2015
static std::mutex contextInitMutexMSVC2013;
#endif

PotentialValue<OpenCLContext> OpenCLContext::getInstance() {
#if (_MSC_VER && _MSC_VER < 1900)
  std::lock_guard<std::mutex> initLock(contextInitMutexMSVC2013);
#endif
  static bool created = false;
  static int deviceID = -1;
  int defaultDevice = -1;
  GPU::getDefaultBackendDevice(&defaultDevice);

  if (defaultDevice < 0) {
    return PotentialValue<OpenCLContext>({Origin::GPU, ErrType::ImplementationError, "[OpenCL] Negative device ID"});
  }
  auto potNumDevices = OpenCLDeviceInfo::getInstance().getNumberOfOpenCLDevices();
  FAIL_RETURN(potNumDevices.status());
  if (defaultDevice >= potNumDevices.value()) {
    return PotentialValue<OpenCLContext>({Origin::GPU, ErrType::ImplementationError,
                                          "[OpenCL] trying to create a context for a device that does not exist"});
  }
  std::lock_guard<std::mutex> lock(contextMutex);
  if ((created && deviceID != defaultDevice) || destroyed) {
    Logger::warning("OpenCL") << "Trying to create an OpenCL context while one has already been created.  "
                                 "This is not supported"
                              << std::endl;
    return PotentialValue<OpenCLContext>(
        {Origin::GPU, ErrType::ImplementationError, "Multiple OpenCL Contexts in one single run is not supported"});
  }
  static OpenCLContext instance(defaultDevice);
  if (!instance.contextStatus.ok()) {
    return PotentialValue<OpenCLContext>(instance.contextStatus);
  }
  created = true;
  deviceID = defaultDevice;
  return PotentialValue<OpenCLContext>(instance);
}

Status printDeviceInfo(int vsDeviceID, cl_device_id device_id) {
  size_t maxWGSize = 0;
  PROPAGATE_CL_ERR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWGSize), &maxWGSize, nullptr));

  cl_uint maxDims = 0;
  PROPAGATE_CL_ERR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxDims), &maxDims, nullptr));

  size_t workItemDims[16];
  PROPAGATE_CL_ERR(
      clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItemDims), &workItemDims, nullptr));

  char version[1024];
  PROPAGATE_CL_ERR(clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(version), &version[0], nullptr));

  Discovery::OpenCLDevice prop;
  if (!(Discovery::getOpenCLDeviceProperties(vsDeviceID, prop))) {
    return {Origin::GPU, ErrType::RuntimeError, "Cannot get Device properties"};
  }

  auto logger = Logger::get(Logger::Info);
  logger << prop.gpuProperties << std::endl;
  logger << "OpenCL version: " << version << std::endl;
  logger << "OpenCL MAX_WORK_GROUP_SIZE: " << maxWGSize << std::endl;
  logger << "OpenCL MAX_WORK_ITEM_SIZES: (";
  for (cl_uint i = 0; i < maxDims; i++) {
    if (i > 0) {
      logger << ", ";
    }
    logger << workItemDims[i];
  }
  logger << ")" << std::endl;

  return Status::OK();
}

void openCLContextNotification(const char* errinfo, const void* /* private_info */, size_t /* callback */,
                               void* /* user_data */) {
  Logger::get(Logger::Error) << "[OpenCL] " << errinfo << std::endl;
}

OpenCLContext::OpenCLContext(int backendDeviceID) : backendDeviceID(backendDeviceID) {
  const auto& info = OpenCLDeviceInfo::getInstance();

  PotentialValue<cl_device_id> potID = info.getOpenCLDeviceID(backendDeviceID);
  PotentialValue<cl_platform_id> potPlatform = info.getPlatform(backendDeviceID);
  if (potID.ok() && potPlatform.ok()) {
    device_id = potID.value();

    int err;
    // Additional attributes to OpenCL context creation
    // which associate an OpenGL context with the OpenCL context
    // http://sa10.idav.ucdavis.edu/docs/sa10-dg-opencl-gl-interop.pdf
#ifdef _WIN32
    cl_platform_id platform = potPlatform.value();
    if (wglGetCurrentContext()) {
      cl_context_properties props[] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platform,  // OpenCL platform
          CL_GL_CONTEXT_KHR,
          (cl_context_properties)wglGetCurrentContext(),  // WGL context
          CL_WGL_HDC_KHR,
          (cl_context_properties)wglGetCurrentDC(),  // HDC used to create the OpenGL context
          0};
      context = clCreateContext(props, 1, &device_id, &openCLContextNotification, nullptr, &err);
    } else {
      context = clCreateContext(0, 1, &device_id, &openCLContextNotification, nullptr, &err);
      Logger::warning("OpenCL") << "Creating context without OpenGL context sharing!" << std::endl;
    }
#elif __ANDROID__
    cl_platform_id platform = potPlatform.value();
    if (eglGetCurrentContext()) {
      cl_context_properties props[] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platform,  // OpenCL platform
          CL_GL_CONTEXT_KHR,
          (cl_context_properties)eglGetCurrentContext(),  // EGL Context
          CL_EGL_DISPLAY_KHR,
          (cl_context_properties)eglGetCurrentDisplay(),  // EGL Display
          0,
      };
      context = clCreateContext(props, 1, &device_id, &openCLContextNotification, nullptr, &err);
    } else {
      context = clCreateContext(0, 1, &device_id, &openCLContextNotification, nullptr, &err);
      Logger::warning("OpenCL") << "Creating context without OpenGL context sharing!" << std::endl;
    }
#elif __linux__
    cl_platform_id platform = potPlatform.value();
    if (glXGetCurrentContext()) {
      cl_context_properties props[] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platform,  // OpenCL platform
          CL_GL_CONTEXT_KHR,
          (cl_context_properties)glXGetCurrentContext(),  // GLX Context
          CL_GLX_DISPLAY_KHR,
          (cl_context_properties)glXGetCurrentDisplay(),  // GLX Display
          0,
      };
      context = clCreateContext(props, 1, &device_id, &openCLContextNotification, nullptr, &err);
    } else {
      context = clCreateContext(0, 1, &device_id, &openCLContextNotification, nullptr, &err);
      Logger::warning("OpenCL") << "Creating context without OpenGL context sharing!" << std::endl;
    }
#elif __APPLE__
    if (CGLGetCurrentContext()) {
      cl_context_properties props[] = {CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                                       (cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()), 0};
      context = clCreateContext(props, 1, &device_id, &openCLContextNotification, nullptr, &err);
    } else {
      context = clCreateContext(0, 1, &device_id, &openCLContextNotification, nullptr, &err);
      Logger::warning("OpenCL") << "Creating context without OpenGL context sharing!" << std::endl;
    }
#endif
    contextStatus = CL_ERROR(err);
    printDeviceInfo(backendDeviceID, device_id);
  } else {
    context = nullptr;
    contextStatus = {Origin::GPU, ErrType::SetupFailure, "Cannot create OpenCL context", potID.status()};
  }
  assert(contextStatus.ok());
};

OpenCLContext::OpenCLContext() {
  context = nullptr;
  contextStatus = {Origin::GPU, ErrType::ImplementationError, "Uninitialized OpenCL context"};
};

CLKernel& OpenCLContext::getKernel(std::string program_name, std::string kernel_name) {
  CLProgram& program = ProgramRegistry::get(program_name);
  if (!program.isCompiled()) {
    program.compile(*this);
  }
  return program.getKernel(kernel_name);
};

PotentialValue<const char*> CLProgram::getSource(const Discovery::DeviceProperties& prop) {
  if (prop.addressBits == 32) {
    return binary32;
  }
  if (prop.addressBits == 64) {
    return binary64;
  }
  return PotentialValue<const char*>(
      {Origin::GPU, ErrType::UnsupportedAction,
       "Cannot compile OpenCL program for address size " + std::to_string(prop.addressBits)});
}

PotentialValue<const char*> CLProgram::getAlternativeSource(const Discovery::DeviceProperties& prop) {
#ifdef ALTERNATIVE_OPENCL_SPIR
  if (prop.addressBits == 32) {
    return alternativeBinary32;
  }
  if (prop.addressBits == 64) {
    return alternativeBinary64;
  }
  return PotentialValue<const char*>(
      {Origin::GPU, ErrType::UnsupportedAction,
       "Cannot compile OpenCL program for address size " + std::to_string(prop.addressBits)});
#else
  (void)prop;
  return PotentialValue<const char*>({Origin::GPU, ErrType::ImplementationError,
                                      "[OpenCL] Libvideostitch_opencl has been built without the alternative SPIR "
                                      "option enabled. Please recompile with this option"});
#endif
}

PotentialValue<size_t> CLProgram::getSourceLength(const Discovery::DeviceProperties& prop) {
  if (prop.addressBits == 32) {
    return binaryLength32;
  }
  if (prop.addressBits == 64) {
    return binaryLength64;
  }
  return PotentialValue<size_t>({Origin::GPU, ErrType::UnsupportedAction,
                                 "Cannot compile OpenCL program for address size " + std::to_string(prop.addressBits)});
}

PotentialValue<size_t> CLProgram::getAlternativeSourceLength(const Discovery::DeviceProperties& prop) {
#ifdef ALTERNATIVE_OPENCL_SPIR
  if (prop.addressBits == 32) {
    return alternativeBinaryLength32;
  }
  if (prop.addressBits == 64) {
    return alternativeBinaryLength64;
  }
  return PotentialValue<size_t>({Origin::GPU, ErrType::UnsupportedAction,
                                 "Cannot compile OpenCL program for address size " + std::to_string(prop.addressBits)});
#else
  (void)prop;
  return PotentialValue<size_t>({Origin::GPU, ErrType::ImplementationError,
                                 "[OpenCL] Libvideostitch_opencl has been built without the alternative SPIR option "
                                 "enabled. Please recompile with this option"});
#endif
}

void CLProgram::searchProgram(OpenCLContext& context, Discovery::DeviceProperties& prop, const cl_device_id& device_id,
                              const char* source, size_t sourceLength) {
  ProgramBinary programCache(programName, source, sourceLength);
  cached = programCache.cacheLoad(program, device_id, prop.name, prop.driverVersion, context);
  // attest that the corresponding binary has been searched for on cache
  searched = true;
}

PotentialValue<size_t> CLProgram::sizeToCompile(OpenCLContext& context) {
  cl_device_id device_id = context.deviceID();
  Discovery::OpenCLDevice prop;
  if (!(Discovery::getOpenCLDeviceProperties(context.get_backendDeviceId(), prop))) {
    return PotentialValue<size_t>({Origin::GPU, ErrType::RuntimeError, "Cannot get Device properties"});
  }
  auto potSource = getSource(prop.gpuProperties);
  FAIL_RETURN(potSource.status());
  const char* source = potSource.value();
  auto potSourceLength = getSourceLength(prop.gpuProperties);
  FAIL_RETURN(potSourceLength.status());
  size_t sourceLength = potSourceLength.value();
  searchProgram(context, prop.gpuProperties, device_id, source, sourceLength);
  if (cached) {
    return 0;
  }
  return getSourceLength(prop.gpuProperties);
}

Status CLProgram::compile(OpenCLContext& context) {
  std::lock_guard<std::mutex> lock(mutex);
  if (isCompiled()) {
    return compilationStatus;
  }
  cl_int err;
  Discovery::OpenCLDevice prop;
  if (!(Discovery::getOpenCLDeviceProperties(context.get_backendDeviceId(), prop))) {
    return {Origin::GPU, ErrType::RuntimeError, "Cannot get Device properties"};
  }
  auto potSource = getSource(prop.gpuProperties);
  FAIL_RETURN(potSource.status());
  const char* source = potSource.value();
  auto potSourceLength = getSourceLength(prop.gpuProperties);
  FAIL_RETURN(potSourceLength.status());
  size_t sourceLength = potSourceLength.value();
  cl_device_id device_id = context.deviceID();
  if (!searched) {
    searchProgram(context, prop.gpuProperties, device_id, source, sourceLength);
  }

  if (!cached) {
#ifdef DISABLE_OPENCL_SPIR
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceLength, &err);
#else  // DISABLE_OPENCL_SPIR
    program =
        clCreateProgramWithBinary(context, 1, &device_id, &sourceLength, (const unsigned char**)&source, nullptr, &err);
#endif
    PROPAGATE_CL_ERR(err);
  }
  const char* flags = "-cl-fast-relaxed-math -cl-mad-enable";
  err = clBuildProgram(program, 1, &device_id, flags, nullptr, nullptr);

#ifdef ALTERNATIVE_OPENCL_SPIR
  if (err != CL_SUCCESS) {
    VideoStitch::Logger::get(VideoStitch::Logger::Info)
        << "[OpenCL] " << programName << " program creation with SPIR failed with the error " << err
        << ". Trying with the alternative SPIR" << std::endl;
    auto potAlternativeSource = getAlternativeSource(prop.gpuProperties);
    FAIL_RETURN(potAlternativeSource.status());
    const char* alternativeSource = potAlternativeSource.value();
    auto potAlternativeSourceLength = getAlternativeSourceLength(prop.gpuProperties);
    FAIL_RETURN(potAlternativeSourceLength.status());
    size_t alternativeSourceLength = potAlternativeSourceLength.value();
    program = clCreateProgramWithBinary(context, 1, &device_id, &alternativeSourceLength,
                                        (const unsigned char**)&alternativeSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device_id, flags, nullptr, nullptr);
  }
#endif  // ALTERNATIVE_OPENCL_SPIR

  if (err != CL_SUCCESS) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error)
        << "OpenCL failed to build program " << programName << std::endl;

    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    char* log = (char*)malloc(log_size);

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
    for (size_t i = 0; i < log_size; ++i) {
      // don't let apps logger choke
      if (i % 256 == 0) {
        VideoStitch::Logger::get(VideoStitch::Logger::Error) << std::flush;
      }
      VideoStitch::Logger::get(VideoStitch::Logger::Error) << log[i];
    }
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << std::endl;
    free(log);

    compilationStatus = {Origin::GPU, ErrType::RuntimeError, "Failed to build program " + programName, CL_ERROR(err)};
    return compilationStatus;
  }
  if (!cached) {
    Discovery::OpenCLDevice prop;
    if (!(Discovery::getOpenCLDeviceProperties(context.get_backendDeviceId(), prop))) {
      return {Origin::GPU, ErrType::RuntimeError, "Cannot get Device properties"};
    }
    ProgramBinary programCache(programName, source, sourceLength);
    programCache.cacheSave(program, prop.gpuProperties.name, prop.gpuProperties.driverVersion);
  }
  compilationStatus = Status::OK();
  compiled = true;

  // TODO_OPENCL_IMPL
  // check status each time

  return VideoStitch::Status::OK();
}

std::vector<std::string> splitStringBySemicolon(std::string source) {
  std::vector<std::string> strings;
  std::string::size_type pos = 0;
  std::string::size_type prev = 0;
  while ((pos = source.find(";", prev)) != std::string::npos) {
    strings.push_back(source.substr(prev, pos - prev));
    prev = pos + 1;
  }
  strings.push_back(source.substr(prev));
  return strings;
};

std::vector<std::string> CLProgram::queryKernelNames() {
  if (isCompiled()) {
    std::vector<char> kernelNames;
    cl_int err;
    {
      std::lock_guard<std::mutex> lock(mutex);
      size_t kernel_names_size;
      err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &kernel_names_size);
      if (err == CL_SUCCESS) {
        kernelNames.resize(kernel_names_size);
        err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, kernelNames.size(), kernelNames.data(), nullptr);
      }
    }
    if (err == CL_SUCCESS) {
      std::string names(kernelNames.begin(), kernelNames.end());
      return splitStringBySemicolon(names);
    }
  }
  return {};
}

/**
 * Compute the total size (number of characters) of the source files we want to compile
 * @param aheadOfTime if true, only consider the programs that are marked as likelyUsed
 */
PotentialValue<size_t> OpenCLContext::computeTotSizeToCompile(const bool aheadOfTime) {
  // reset total size value
  size_t totalSizeToCompile = 0;
  auto& programs = ProgramRegistry::programRegister();
  for (auto it = programs.begin(); it != programs.end(); ++it) {
    CLProgram& program = it->second;

    if (!program.isCompiled() && (!(aheadOfTime) || program.isLikelyUsed())) {
      auto potSizeToCompile = program.sizeToCompile(*this);
      FAIL_RETURN(potSizeToCompile.status());
      totalSizeToCompile += potSizeToCompile.value();
    }
  }
  return PotentialValue<size_t>(totalSizeToCompile);
}
/**
 * Compile all the OpenCL programs.
 * @param progress if non-null, used as progress indicator.
 * @param aheadOfTime if true, only compile the programs that are marked as likelyUsed
 */
Status OpenCLContext::compileAllPrograms(Util::Algorithm::ProgressReporter* pReporter, const bool aheadOfTime) {
  Discovery::OpenCLDevice prop;
  if (!(Discovery::getOpenCLDeviceProperties(backendDeviceID, prop))) {
    return {Origin::GPU, ErrType::RuntimeError, "Cannot get Device properties"};
  }
  size_t sizeCompiled = 0;
  PotentialValue<size_t> potSizeToCompile = computeTotSizeToCompile(aheadOfTime);
  size_t totalSizeToCompile;
  if (potSizeToCompile.ok()) {
    totalSizeToCompile = potSizeToCompile.value();
  } else {
    return potSizeToCompile.status();
  }
  if (totalSizeToCompile && pReporter) {
    bool shouldCancel =
        pReporter->notify("Optimizing the application for your hardware. This may take several minutes...", 0.0);
    if (shouldCancel) {
      return Status{Origin::GPU, ErrType::OperationAbortedByUser, "OpenCL compilation cancelled by the user."};
    }
  }
  auto& programs = ProgramRegistry::programRegister();
  for (auto it = programs.begin(); it != programs.end(); ++it) {
    const std::string& programName = it->first;
    CLProgram& program = it->second;

    if (!program.isCompiled() && (!(aheadOfTime) || program.isLikelyUsed())) {
      VideoStitch::Logger::get(VideoStitch::Logger::Info) << "OpenCL building program " + programName + "...\n";
      FAIL_RETURN(program.compile(*this));
      if (!program.isCached()) {
        assert(totalSizeToCompile != 0);
        PotentialValue<size_t> potSourceLength = program.getSourceLength(prop.gpuProperties);
        FAIL_RETURN(potSourceLength.status());
        sizeCompiled += potSourceLength.value();
        if (pReporter) {
          bool shouldCancel =
              pReporter->notify("Optimizing the application for your hardware. This may take several minutes...",
                                (double)sizeCompiled / totalSizeToCompile);
          if (shouldCancel) {
            return Status{Origin::GPU, ErrType::OperationAbortedByUser, "OpenCL compilation cancelled by the user."};
          }
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace GPU
}  // namespace VideoStitch
