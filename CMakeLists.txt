# Supported CMAKE_BUILD_TYPE configurations:
# - 'Debug': all platforms
# - 'Release': all platforms
# - 'RelWithDebInfo': Linux & Apple
# - 'MinSizeRel': Linux & Apple
# - 'ASan' (address sanitizer): compiler dependent, sets HAVE_ADDRESS_SANITIZER
# - 'LSan' (leak sanitizer): compiler dependent, sets HAVE_LEAK_SANITIZER
# - 'TSan' (thread sanitizer): compiler dependent, sets HAVE_THREAD_SANITIZER
# - 'UBSan' (undefined behavior sanitizer): compiler dependent, sets HAVE_UNDEFINED_BEHAVIOR_SANITIZER

cmake_minimum_required(VERSION 3.11)
project(VideoStitch)

option(BUILD_LIB "Build VideoStitch lib" ON)
option(BUILD_STATIC_LIB "Build static VideoStitch lib" OFF)
option(BUILD_IO_PLUGINS "Build VideoStitch I/O plugins" ON)
option(BUILD_APPS "Build VideoStitch applications" ON)

option(BUILD_MACOSX_BUNDLE "Build the mac package instead of the binaries" OFF)

option(LCOV "Build apps with coverage" OFF)
option(STAGING "Test staging (test developer option)" OFF)

option(NO_CCACHE "Do not use ccache" OFF)

option(IWYU_ENABLED "Compile using include-what-you-use" OFF)
option(DISABLE_EXR  "Create OpenEXR I/O plugin" ${ANDROID})

# ----------------------------------------------------------------------------
# GPU backend options
# ----------------------------------------------------------------------------
option(GPU_BACKEND_CUDA "Build with CUDA backend" ON)
option(GPU_BACKEND_OPENCL "Build with OpenCL backend" OFF)
message(STATUS "GPU_BACKEND_CUDA='${GPU_BACKEND_CUDA}'")
message(STATUS "GPU_BACKEND_OPENCL='${GPU_BACKEND_OPENCL}'")

set(GPU_BACKEND_DEFAULT "CUDA" CACHE STRING "Default GPU backend")
set_property(CACHE GPU_BACKEND_DEFAULT PROPERTY STRINGS OPENCL CUDA)

if(NOT GPU_BACKEND_CUDA AND NOT GPU_BACKEND_OPENCL)
  message(FATAL_ERROR "Please select at least one backend")
endif()

# Force default backend if only one is selected
if(GPU_BACKEND_CUDA AND NOT GPU_BACKEND_OPENCL)
  set(GPU_BACKEND_DEFAULT "CUDA" CACHE STRING "Default GPU backend" FORCE)
elseif(GPU_BACKEND_OPENCL AND NOT GPU_BACKEND_CUDA)
  set(GPU_BACKEND_DEFAULT "OPENCL" CACHE STRING "Default GPU backend" FORCE)
endif()

# ----------------------------------------------------------------------------
# Include used all over the project
# ----------------------------------------------------------------------------
set(TESTING_INCLUDE ${CMAKE_SOURCE_DIR}/tests/include)

# ----------------------------------------------------------------------------
# Auto-select bitness based on platform
# ----------------------------------------------------------------------------
if( NOT BITNESS )
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BITNESS 64)
    else()
        set(BITNESS 32)
    endif()
endif()

if( BITNESS EQUAL 64 )
  set(BITNESS_SUFFIX x86_64)
elseif( BITNESS EQUAL 32 )
  set(BITNESS_SUFFIX x86)
else()
  message( FATAL_ERROR "Bitness specified is invalid" )
endif()

# provide safeguard for subdirectories
set(VIDEOSTITCH_CMAKE YES)

if( NOT NO_CCACHE )
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    message(STATUS "Compiler caching enabled (ccache)")
    set(CCACHE_ENABLED ON)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)
endif( NOT NO_CCACHE )

# ----------------------------------------------------------------------------
# Determine current platform
# ----------------------------------------------------------------------------
if(APPLE)
  message(STATUS "System: APPLE")
  option(MACPORTS "Use MacPorts, OFF uses Homebrew" ON)
endif(APPLE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
  set(ANDROID True)
  message(STATUS "System: ANDROID")
else(${CMAKE_SYSTEM_NAME} MATCHES "Android")
  set(NANDROID True)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Android")

if(UNIX AND NOT APPLE AND NOT ANDROID)
  set(LINUX True)
  set(PLATFORM_SHORTHAND lnx)
  message(STATUS "System: LINUX")
endif(UNIX AND NOT APPLE AND NOT ANDROID)

if(WIN32)
  set(WINDOWS True)
  set(PLATFORM_SHORTHAND win)
  message(STATUS "System: WINDOWS")

  if (WIN_CHOCO)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
      message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
      file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.14/conan.cmake"
                    "${CMAKE_BINARY_DIR}/conan.cmake")
    endif()
    include(${CMAKE_BINARY_DIR}/conan.cmake)

    conan_add_remote(NAME bincrafters URL https://api.bintray.com/conan/bincrafters/public-conan)
    conan_cmake_run(REQUIRES
                             eigen/3.3.7@conan/stable
                             libpng/1.6.37@bincrafters/stable
                             glew/2.1.0@bincrafters/stable
                             glfw/3.3@bincrafters/stable
                             # ceres-solver/1.12.0@jmdaly/testing
                    BUILD missing
                    )

    message(STATUS "CONAN_LIBS: ${CONAN_LIBS}")
    message(STATUS "CONAN_LIBS_LIBPNG: ${CONAN_LIBS_LIBPNG}")
    message(STATUS "CONAN_LIBS_GLEW: ${CONAN_LIBS_GLEW}")
    message(STATUS "CONAN_LIBS_GLFW: ${CONAN_LIBS_GLFW}")

    message(STATUS "CONAN_INCLUDE_DIRS: ${CONAN_INCLUDE_DIRS}")
    message(STATUS "CONAN_INCLUDE_DIRS_EIGEN: ${CONAN_INCLUDE_DIRS_EIGEN}")


  endif()

endif(WIN32)

# ----------------------------------------------------------------------------
# Determine compiler
# ----------------------------------------------------------------------------
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
  set(COMPILER_CLANG YES)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(COMPILER_GCC YES)
endif()


# ----------------------------------------------------------------------------
# Use definitions from the cmake folder for find_package
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


# ----------------------------------------------------------------------------
# Default build types if not specified by the user
# ----------------------------------------------------------------------------
if(CMAKE_CONFIGURATION_TYPES)
  # Multi-configuration generator
  if(WINDOWS)
    # external deps only exist for Debug and Release
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Build configurations" FORCE)
  endif(WINDOWS)
  message(STATUS "CMake Configuration Types -- ${CMAKE_CONFIGURATION_TYPES}")
else()
  # Single-configuration generator
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
  endif(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMake Build Type -- ${CMAKE_BUILD_TYPE}")
endif(CMAKE_CONFIGURATION_TYPES)


# ----------------------------------------------------------------------------
# Enable grouping targets into folders in IDEs like Visual Studio
# ----------------------------------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# ----------------------------------------------------------------------------
# Platform dependent names for lib and output directories
# ----------------------------------------------------------------------------
if(WINDOWS)
  set(VS_LIB_CUDA libvideostitch_cuda)
  set(VS_LIB_OPENCL libvideostitch_opencl)
  set(VS_DISCOVERY libvideostitch-gpudiscovery)
else(WINDOWS)
  set(VS_LIB_CUDA videostitch_cuda)
  set(VS_LIB_OPENCL videostitch_opencl)
  set(VS_DISCOVERY videostitch-gpudiscovery)
endif(WINDOWS)

set(VS_LIB_DEFAULT ${VS_LIB_${GPU_BACKEND_DEFAULT}})
set(VS_LIB_STATIC ${VS_LIB_DEFAULT}_static)

if(WINDOWS)
  set(BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
  set(VS_LIB_UNIT_TEST ${VS_LIB_STATIC})
else()
  set(VS_LIB_UNIT_TEST ${VS_LIB_DEFAULT})
endif()

if(APPLE)
  # name differs on OS X to adhere to package naming scheme
  set(VS_PLUGIN_DIR_NAME CorePlugins)
  # VSA-4697 facilitate plugin loading on Mac
  set(VS_STUDIO_DIR_NAME Studio)
  set(VS_VAHANA_DIR_NAME VahanaVR)
  set(VS_VAHANA_PLUGIN_DIR_NAME VahanaPlugins)
else()
  set(VS_PLUGIN_DIR_NAME core_plugins)
  set(VS_STUDIO_DIR_NAME )
  set(VS_VAHANA_DIR_NAME )
  set(VS_VAHANA_PLUGIN_DIR_NAME vahana_plugins)
endif()

set(VS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/x64)

if (WINDOWS)
  # All files will be built in the same folder that external deps were downloaded.
  set(VS_OUT_DIR ${PROJECT_SOURCE_DIR}/bin/x64)
else(WINDOWS)
  set(VS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/x64)
endif(WINDOWS)

set(VS_TEST_DIR_NAME Test)

set(VS_LIB_PUBLIC_HEADERS_DIR "${PROJECT_SOURCE_DIR}/lib/include")
set(VS_DISCOVERY_PUBLIC_HEADERS_DIR "${PROJECT_SOURCE_DIR}/discovery/include")
set(GENERATED_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/generated_headers")
file(MAKE_DIRECTORY ${GENERATED_HEADERS})
set(VERSION_HEADER ${GENERATED_HEADERS}/version.hpp)

if(NOT CMAKE_EXTERNAL_DEPS)
  set(CMAKE_EXTERNAL_DEPS "${PROJECT_SOURCE_DIR}/external_deps")
endif(NOT CMAKE_EXTERNAL_DEPS)
set(CMAKE_EXTERNAL_LIB "${CMAKE_EXTERNAL_DEPS}/lib")
set(CMAKE_EXTERNAL_INCLUDE "${CMAKE_EXTERNAL_DEPS}/include")

# ----------------------------------------------------------------------------
# Set binary directory structure manually
# to get bin/Debug/CorePlugins/libav.dylib instead of bin/CorePlugins/Debug/libav.dylib
# ----------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE)
  # Set output dir for the generic no-config case (e.g. make, ninja)
  string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOW)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW})

  set(VS_PLUGIN_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW}/${VS_PLUGIN_DIR_NAME})
  set(VS_VAHANA_PLUGIN_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW}/${VS_VAHANA_PLUGIN_DIR_NAME})
  set(VS_TEST_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW})
  set(VS_STUDIO_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW}/${VS_STUDIO_DIR_NAME})
  set(VS_VAHANA_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW}/${VS_VAHANA_DIR_NAME})
endif(CMAKE_BUILD_TYPE)

# Set output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
  string(TOLOWER ${OUTPUTCONFIG} OUTPUTCONFIG_LOW)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW})

  set(VS_PLUGIN_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_PLUGIN_DIR_NAME})
  set(VS_STUDIO_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_STUDIO_DIR_NAME})
  set(VS_VAHANA_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_VAHANA_DIR_NAME})
  set(VS_VAHANA_PLUGIN_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_VAHANA_PLUGIN_DIR_NAME})
  set(VS_TEST_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# ----------------------------------------------------------------------------
# Global compilation flags
# ----------------------------------------------------------------------------
include(FindASan)
include(FindLSan)
include(FindTSan)
include(FindUBSan)
include(FindIWYU)

include(CppcheckTargets)

if((CMAKE_BUILD_TYPE STREQUAL "ASan") AND NOT HAVE_ADDRESS_SANITIZER)
  message(FATAL_ERROR "Address sanitizer not supported by compiler!")
endif()

if((CMAKE_BUILD_TYPE STREQUAL "LSan") AND NOT HAVE_LEAK_SANITIZER)
  message(FATAL_ERROR "Leak sanitizer not supported by compiler!")
endif()

if((CMAKE_BUILD_TYPE STREQUAL "TSan") AND NOT HAVE_THREAD_SANITIZER)
  message(FATAL_ERROR "Thread sanitizer not supported by compiler!")
endif()

if((CMAKE_BUILD_TYPE STREQUAL "UBSan") AND NOT HAVE_UNDEFINED_BEHAVIOR_SANITIZER)
  message(FATAL_ERROR "Undefined behavior sanitizer not supported by compiler!")
endif()

if(IWYU_ENABLED AND NOT HAVE_IWYU)
  message(FATAL_ERROR "Include-what-you-use not supported by compiler!")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "TSan")
  # isolate problems found at exit, when destroying static objects of libvideostitch
  # static linking will remove unused objects
  set(BUILD_STATIC_LIB ON)
  set(VS_LIB_UNIT_TEST ${VS_LIB_STATIC})
endif()

if(LINUX AND LCOV)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif(LINUX AND LCOV)

if(MSVC)
  # C4996: Unsafe function: FIXME
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /errorReport:prompt /GS /MP /WX- /Zc:forScope /Zc:wchar_t /wd4996")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4099")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4099 /OPT:REF /OPT:ICF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
  if(LCOV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
  endif()
elseif(COMPILER_CLANG)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wrange-loop-analysis -fcolor-diagnostics -Wno-sign-conversion")

  # as of 02/2017 macbb (Xcode 8.2/clang) warns about code
  # that executes after assert(false)
  if((NOT CMAKE_BUILD_TYPE) OR (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Wunreachable-code")
  endif()
elseif(COMPILER_GCC)
  set(CMAKE_CXX_FLAGS
      # missing-field-initializers : very had to fix, it changed after c++11
      "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-field-initializers")
endif()

# ----------------------------------------------------------------------------
# CMake helper macros
# ----------------------------------------------------------------------------
function(include_lib_vs_headers target)
    target_include_directories(${target} PRIVATE ${VS_LIB_PUBLIC_HEADERS_DIR})
    target_include_directories(${target} PRIVATE ${GENERATED_HEADERS})
endfunction(include_lib_vs_headers target)

function(include_discovery_vs_headers target)
  target_include_directories(${target} PRIVATE ${VS_DISCOVERY_PUBLIC_HEADERS_DIR})
endfunction(include_discovery_vs_headers target)

function(set_iwyu target)
  if(IWYU_ENABLED)
    set_property(TARGET ${target} PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${IWYU})
  endif()
endfunction(set_iwyu target)

# ----------------------------------------------------------------------------
# Define a grouping for source files in IDE project generation
# ----------------------------------------------------------------------------
function(assign_source_group)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE ${_source})
      file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
    else()
      set(_source_rel ${_source})
    endif()
    get_filename_component(_source_path "${_source_rel}" DIRECTORY)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

# ----------------------------------------------------------------------------
# A function to find debug and optimized library
# ----------------------------------------------------------------------------
if (WINDOWS)
  function(find_debug_and_optimized_library LIBRARY_NAME LIBRARY_PATH_DEBUG NAME_DEBUG LIBRARY_PATH_OPTIMIZED NAME_OPTIMIZED)
    find_library(${LIBRARY_NAME}_DEBUG NAMES ${NAME_DEBUG} PATHS "${CMAKE_EXTERNAL_LIB}/${LIBRARY_PATH_DEBUG}" NO_DEFAULT_PATH)
    find_library(${LIBRARY_NAME}_OPTIMIZED NAMES ${NAME_OPTIMIZED} PATHS "${CMAKE_EXTERNAL_LIB}/${LIBRARY_PATH_OPTIMIZED}" NO_DEFAULT_PATH)
    set(${LIBRARY_NAME}
      debug ${${LIBRARY_NAME}_DEBUG}
      optimized ${${LIBRARY_NAME}_OPTIMIZED}
      PARENT_SCOPE)
  endfunction()
endif(WINDOWS)

# ----------------------------------------------------------------------------
# Find libs that we need for mutiple targets and add astyle target
# ----------------------------------------------------------------------------
include(external_deps)
include(astyle)

# ----------------------------------------------------------------------------

enable_testing()

# ----------------------------------------------------------------------------
# CUDA
# We need it here, since rtmp plugin needs it
# ----------------------------------------------------------------------------
if(GPU_BACKEND_CUDA)
  include(CUDA)
endif(GPU_BACKEND_CUDA)

# ----------------------------------------------------------------------------
# Delay load on Windows
# ----------------------------------------------------------------------------
set(USE_DELAY_LOAD "OFF")
if(GPU_BACKEND_CUDA AND GPU_BACKEND_OPENCL AND MSVC)
  set(USE_DELAY_LOAD "ON")
endif()
message(STATUS "USE_DELAY_LOAD = ${USE_DELAY_LOAD}")

if(USE_DELAY_LOAD)
  find_library(DELAY_LOAD_LIB NAMES "DelayImp.lib")
endif()

if(WINDOWS)
  set(VS_LIB_FAKE libvideostitch)
else()
  set(VS_LIB_FAKE videostitch)
endif(WINDOWS)
if(CMAKE_BUILD_TYPE)
  add_custom_target(generateFakeLibvideostitch
                    COMMAND ${CMAKE_COMMAND}
                    -DVS_LIB_FAKE=${VS_LIB_FAKE}
                    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                    -P ${CMAKE_SOURCE_DIR}/lib/generateFakeLibvideostitch.cmake
                    DEPENDS ${VS_LIB_DEFAULT})
  if(USE_DELAY_LOAD)
    add_dependencies(generateFakeLibvideostitch ${VS_LIB_FAKE})
  endif()
else()
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
    add_custom_target(generateFakeLibvideostitch_${OUTPUTCONFIG_UP}
                      COMMAND ${CMAKE_COMMAND}
                      -DVS_LIB_FAKE=${VS_LIB_FAKE}
                      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP}}
                      -P ${CMAKE_SOURCE_DIR}/lib/generateFakeLibvideostitch.cmake
                      DEPENDS ${VS_LIB_DEFAULT})
    if(USE_DELAY_LOAD)
      add_dependencies(generateFakeLibvideostitch_${OUTPUTCONFIG_UP} ${VS_LIB_FAKE})
    endif()
  endforeach()
endif()

# This function should be use by every dll/exe that links on libvideostitch
function(link_target_to_libvideostitch TARGET_NAME)
  if(WINDOWS)
    if(CMAKE_BUILD_TYPE)
      add_dependencies(${TARGET_NAME} generateFakeLibvideostitch)
    else()
      foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
        add_dependencies(${TARGET_NAME} generateFakeLibvideostitch_${OUTPUTCONFIG_UP})
      endforeach()
    endif(CMAKE_BUILD_TYPE)
  endif(WINDOWS)
  if(USE_DELAY_LOAD)
    target_link_libraries(${TARGET_NAME} PRIVATE ${DELAY_LOAD_LIB})
    target_link_libraries(${TARGET_NAME} PRIVATE ${VS_LIB_FAKE})
    target_compile_definitions(${TARGET_NAME} PRIVATE DELAY_LOAD_ENABLED=1)
    set(new_link_flags "/DELAYLOAD:${VS_LIB_FAKE}.dll")
    get_target_property(existing_link_flags ${TARGET_NAME} LINK_FLAGS)
    if(existing_link_flags)
      set(new_link_flags "${existing_link_flags} ${new_link_flags}")
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS ${new_link_flags})
  elseif(APPLE)
    target_link_libraries(${TARGET_NAME} PRIVATE ${VS_LIB_FAKE})
    # ensure the non-fake backend libs are built
    if(GPU_BACKEND_CUDA)
      add_dependencies(${TARGET_NAME} ${VS_LIB_CUDA})
    endif()
    if(GPU_BACKEND_OPENCL)
      add_dependencies(${TARGET_NAME} ${VS_LIB_OPENCL})
    endif()
  else()
    target_link_libraries(${TARGET_NAME} PUBLIC ${VS_LIB_DEFAULT})
  endif(USE_DELAY_LOAD)
endfunction()


# ----------------------------------------------------------------------------
# Android specific after android.toolchain.cmake
# ----------------------------------------------------------------------------
if(ANDROID)
  include(cmake/android.cmake)
endif(ANDROID)

# ----------------------------------------------------------------------------
# Build libGPUDiscovery
# ----------------------------------------------------------------------------
add_subdirectory(discovery)

# ----------------------------------------------------------------------------
# Build libvideostitch
# ----------------------------------------------------------------------------
if(BUILD_STATIC_LIB AND (NOT BUILD_LIB))
  message(FATAL_ERROR "BUILD_STATIC_LIB implies BUILD_LIB")
endif()

if(BUILD_LIB)
  add_subdirectory(lib)
endif (BUILD_LIB)


# ----------------------------------------------------------------------------
# Build I/O Plugins
# ----------------------------------------------------------------------------
if(BUILD_IO_PLUGINS)
  add_subdirectory(IO)
  MESSAGE(STATUS "VideoStitch I/O libs configured to build: ${VS_IO_LIBRARIES}")
endif (BUILD_IO_PLUGINS)

macro(add_vs_IO_deps target)
  if(BUILD_IO_PLUGINS)
    add_dependencies(${target} ${VS_IO_LIBRARIES})
  endif(BUILD_IO_PLUGINS)
endmacro()

# ----------------------------------------------------------------------------
# Build apps
# ----------------------------------------------------------------------------
if(BUILD_APPS)
  add_subdirectory(apps)
elseif(ANDROID)
  add_subdirectory(apps/src/AndroidDemo)
endif(BUILD_APPS)


# ----------------------------------------------------------------------------
# Command line and squish tests
# ----------------------------------------------------------------------------
add_subdirectory(tests)


# ----------------------------------------------------------------------------
# Install targets
# ----------------------------------------------------------------------------
if(APPLE AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  install(
    PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/videostitch-cmd
    DESTINATION ${VS_STUDIO_DIR}
  )
endif(APPLE AND CMAKE_RUNTIME_OUTPUT_DIRECTORY)

install(DIRECTORY ${VS_OUT_DIR}
        DESTINATION ${CMAKE_SOURCE_DIR}/bin
        USE_SOURCE_PERMISSIONS)

