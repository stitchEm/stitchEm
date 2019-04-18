# ----------------------------------------------------------------------------
# CUDA backend options
# ----------------------------------------------------------------------------

option(FASTMATH "Build with cuda fast_math" ON)
option(CUDA_KERNEL_PROFILING "Compile to enable additional profiling information" OFF)
option(CUDA_LOCAL_ARCH_ONLY "Compile CUDA kernels only for the local CUDA architecture" OFF)

include (${CMAKE_SOURCE_DIR}/cmake/getCuda.cmake)

if(NOT CUDA_FOUND)
  MESSAGE (ERROR_FATAL "Required cuda version not found !")
endif (NOT CUDA_FOUND)


# ----------------------------------------------------------------------------
# Find CUDA libraries and dependencies
# ----------------------------------------------------------------------------

# TODO: extract find_library + find_package

if (WINDOWS)
  find_library(GLEW NAMES glew32s PATHS ${CMAKE_EXTERNAL_DEPS}/lib/GL NO_DEFAULT_PATH)
  include_directories(${OPENGL_INCLUDE_DIRS})
  include_directories(${OPENCV_INCLUDE_DIRS})
  find_library(CUDA cuda PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)
  find_library(CUDART cudart PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)
  find_library(CUVID nvcuvid PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)
  find_library(NVML nvml PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" NO_DEFAULT_PATH)
endif(WINDOWS)

if (LINUX)
  set(CUDA_PROPAGATE_HOST_FLAGS "FALSE")
  if(TEGRA_DEMO)
    set(LINUX_CUDA_PATH ${CUDA_TOOLKIT_TARGET_DIR}/lib)
    find_library(GLEW_LIBRARIES GLEW NO_CMAKE_FIND_ROOT_PATH)
    find_library(OPENGL_LIBRARIES GL NO_CMAKE_FIND_ROOT_PATH)
  else(TEGRA_DEMO)
    find_package(GLEW)
    find_library(CUDA cuda)
    find_library(CUDART cudart PATHS ${LINUX_CUDA_PATH} NO_DEFAULT_PATH)
    find_library(NVML nvidia-ml)
  endif(TEGRA_DEMO)
  include_directories(${CMAKE_EXTERNAL_DEPS}/include)
  link_directories(${LINUX_CUDA_PATH})
  find_library(CUVID nvcuvid)
endif(LINUX)

if(APPLE)
  find_package(GLEW)
  find_library(CUDA cuda PATHS ${MAC_CUDA_PATH} NO_DEFAULT_PATH)
  find_library(CUDART cudart PATHS ${MAC_CUDA_PATH} NO_DEFAULT_PATH)
  string(REPLACE "/usr/local/cuda/" "${CUDA_TOOLKIT_ROOT_DIR}/" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
endif(APPLE)

message(STATUS "CUDA: ${CUDA}")
message(STATUS "CUDART: ${CUDART}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# ----------------------------------------------------------------------------
# CUDA compilation options
# ----------------------------------------------------------------------------

if(FASTMATH)
  list(APPEND CUDA_NVCC_FLAGS --use_fast_math)
endif(FASTMATH)

if(CUDA_KERNEL_PROFILING)
  add_definitions(-DUSE_NVTX)
  find_library(NVTX nvToolsExt PATHS ${LINUX_CUDA_PATH} NO_DEFAULT_PATH)
  message(STATUS "NVTX: ${NVTX}")
  list(APPEND CUDA_NVCC_FLAGS -lineinfo)
else(CUDA_KERNEL_PROFILING)
  # profiling kernel details failed with default-stream per-thread
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
endif(CUDA_KERNEL_PROFILING)

# ----------------------------------------------------------------------------
# CUDA compilation OS specific flags
# ----------------------------------------------------------------------------

if(LINUX OR ANDROID)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      --compiler-options -fno-strict-aliasing)
endif(LINUX OR ANDROID)

if(LINUX OR APPLE)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -Xcompiler -fPIC)
  if(${CUDA_VERSION} VERSION_LESS 8.0)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Xcompiler -Wno-conversion)
  endif()
endif(LINUX OR APPLE)

if(COMPILER_CLANG)
  # nvcc/clang complains about code after assert(false)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -Xcompiler -Wno-unreachable-code)
endif()


if(WINDOWS)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
       -D_USE_MATH_DEFINES
       -DVS_LIB_COMPILATION
       -DNOMINMAX
       -Xcompiler /FS)
else(WINDOWS)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -std=c++11)
endif(WINDOWS)

if(ANDROID)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -ccbin ${CMAKE_CXX_COMPILER} -O3)
endif(ANDROID)

include(CUDAArch)

# ----------------------------------------------------------------------------
# Determine CUDA architectures to generate code for
# ----------------------------------------------------------------------------

if(CUDA_LOCAL_ARCH_ONLY OR CUDA_TARGET_ARCH)
  # queried and set in CUDAArch
  set(CUDA_NVCC_ARCH_FLAGS ${CUDA_LOCAL_ARCH_FLAGS})
else()
  if(TEGRA_DEMO)
    set(CUDA_NVCC_ARCH_FLAGS
        -gencode=arch=compute_53,code=sm_53
        -gencode=arch=compute_62,code=sm_62)
  else(TEGRA_DEMO)
    set(CUDA_NVCC_ARCH_FLAGS
        -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_61,code=compute_61)
  endif(TEGRA_DEMO)
endif(CUDA_LOCAL_ARCH_ONLY OR CUDA_TARGET_ARCH)

message(STATUS "NVCC arch flags: ${CUDA_NVCC_ARCH_FLAGS}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_ARCH_FLAGS})

