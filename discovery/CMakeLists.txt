project(lib_gpuDiscovery)

# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

option(OCLGRIND "Build for testing with the Oclgrind OpenCL device simulator and debugger" OFF)

if(OCLGRIND AND NOT DISABLE_OPENCL_SPIR)
  message(FATAL_ERROR "OCLGRIND implies DISABLE_OPENCL_SPIR, as Oclgrind does not ingest SPIR")
endif()

if(OCLGRIND AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
  message(FATAL_ERROR "Oclgrind builds are intended for internal testing only.\n"
                      "`Release` builds are not supported as a safeguard against uninteded distribution of broken builds.\n"
                      "If compiler optimizations are required, use RelWithDebInfo locally.")
endif()

set(SOURCES
    src/device.cpp
    src/deviceHelper.cpp
    src/backendLib/backendLibHelper.cpp
    src/backendLib/backendLibLoader.cpp
    src/utils/fileHelper.cpp)

set(HEADERS
    src/device.hpp
    src/deviceHelper.hpp
    src/backendLib/backendLibLoader.hpp
    src/include/backendLibHelper.hpp
    src/include/config.hpp
    src/include/cudaDeviceInfo.hpp
    src/include/delayLoad.hpp
    src/include/fileHelper.hpp
    src/include/genericDeviceInfo.hpp
    src/include/openCLDeviceInfo.hpp)

assign_source_group(${SOURCES} ${HEADERS})

add_library(${VS_DISCOVERY} SHARED ${SOURCES})
add_cppcheck(${VS_DISCOVERY} VS)
include_discovery_vs_headers(${VS_DISCOVERY})

# ----------------------------------------------------------------------------
# OpenCL
# ----------------------------------------------------------------------------

if(NOT ANDROID)
  find_package(OpenCL)
endif()
if(OpenCL_FOUND AND GPU_BACKEND_OPENCL)
  target_include_directories(${VS_DISCOVERY} PRIVATE ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(${VS_DISCOVERY} PUBLIC ${OpenCL_LIBRARIES})
  if(WINDOWS)
    target_link_libraries(${VS_DISCOVERY} PRIVATE "DelayImp.lib")
    set_target_properties(${VS_DISCOVERY} PROPERTIES LINK_FLAGS "/DELAYLOAD:OpenCL.dll")
  endif()
  target_compile_definitions(${VS_DISCOVERY} PRIVATE OPENCL_FOUND)
endif()

# ----------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------

if(NOT CUDA_FOUND)
  include (${CMAKE_SOURCE_DIR}/cmake/getCuda.cmake)
endif (NOT CUDA_FOUND)
if(CUDA_FOUND)
  target_include_directories(${VS_DISCOVERY} PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${VS_DISCOVERY} PRIVATE ${CUDA_LIBRARIES})
  target_compile_definitions(${VS_DISCOVERY} PRIVATE CUDA_FOUND)
endif (CUDA_FOUND)

if(TEGRA_DEMO)
  target_link_libraries(${VS_DISCOVERY} PRIVATE dl)
endif(TEGRA_DEMO)

# ----------------------------------------------------------------------------
# APPLE CORE FOUNDATION
# ----------------------------------------------------------------------------
if(APPLE)
  if(NOT CORE_FOUNDATION)
      find_library(CORE_FOUNDATION CoreFoundation REQUIRED)
  endif(NOT CORE_FOUNDATION)
  target_link_libraries(${VS_DISCOVERY} PRIVATE ${CORE_FOUNDATION})
endif(APPLE)

# ----------------------------------------------------------------------------
# Add compile definitions
# ----------------------------------------------------------------------------
if(OCLGRIND)
  message(STATUS "Building for testing with Oclgrind. OpenCL device list may contain unusable hardware devices.")
  target_compile_definitions(${VS_DISCOVERY} PRIVATE "OCLGRIND")
endif()

# ----------------------------------------------------------------------------
# Configure libdiscovery
# ----------------------------------------------------------------------------

set_property(TARGET ${VS_DISCOVERY} PROPERTY CXX_STANDARD 14)
target_compile_definitions(${VS_DISCOVERY} PRIVATE VS_LIB_DISCOVERY)
set (DISCOVERY_SRC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_discovery_vs_headers(${VS_DISCOVERY})

if(USE_DELAY_LOAD)
  if(MSVC)
    target_compile_definitions(${VS_DISCOVERY} PRIVATE DELAY_LOAD_ENABLED=1)
  elseif(APPLE)
    target_compile_definitions(${VS_DISCOVERY} PRIVATE SYMLINK_UPDATE_ENABLED=1)
  endif()
endif()




# ----------------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------------
set(DEVICE_NAME_TEST DeviceNameTest)

add_executable(${DEVICE_NAME_TEST} src/tests/deviceNameTest.cpp src/deviceHelper.cpp)

set_property(TARGET ${DEVICE_NAME_TEST} PROPERTY CXX_STANDARD 14)

target_include_directories(${DEVICE_NAME_TEST} PRIVATE ${TESTING_INCLUDE})
target_include_directories(${DEVICE_NAME_TEST} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_discovery_vs_headers(${DEVICE_NAME_TEST})

find_package(Threads REQUIRED)

if(NOT APPLE AND NOT ANDROID)
  target_link_libraries(${DEVICE_NAME_TEST} PRIVATE Threads::Threads)
endif()

if(STAGING)
  # Add new tests here
else()
  add_test(NAME ${DEVICE_NAME_TEST} COMMAND ${DEVICE_NAME_TEST})
endif()
