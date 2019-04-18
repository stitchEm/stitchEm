project(VideoStitch-apps)

option(BUILD_STUDIO "Build VideoStitch-Studio" ON)
option(BUILD_APPS_TESTS "Build apps tests" ON)

option(BUILD_VAHANA "Build VahanaVR" ON)
option(ENABLE_YOUTUBE_OUTPUT "Build YouTube output and link on YouTube dependencies" OFF)

# safeguard against accidental misuse
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

# ----------------------------------------------------------------------------
# Find Qt
# ----------------------------------------------------------------------------

file(STRINGS "${PROJECT_SOURCE_DIR}/qt.version" QT_MINI_VERSION)
message(STATUS "Qt minimum supported version: ${QT_MINI_VERSION}")

find_package(Qt5 ${QT_MINI_VERSION} COMPONENTS Core Concurrent Gui Multimedia Network OpenGL Widgets LinguistTools Test REQUIRED)

message(STATUS "Qt FOUND: ${Qt5Core_VERSION}")

# ----------------------------------------------------------------------------

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Let AUTOMOC and AUTOUIC process GENERATED files.
if (POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()

# set(CMAKE_AUTOUIC ON)


include(src/appsversion.cmake)

if(APPLE AND CCACHE_ENABLED AND (Qt5Core_VERSION VERSION_LESS "5.5"))
  # suppress warning in newer clang, don't trigger warning about unknown flag in older clang
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-Wno-inconsistent-missing-override" HAVE_WARNING_INCONSISTENT_MISSING_OVERRIDE)
  if(HAVE_WARNING_INCONSISTENT_MISSING_OVERRIDE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
  endif(HAVE_WARNING_INCONSISTENT_MISSING_OVERRIDE)
endif(APPLE AND CCACHE_ENABLED AND (Qt5Core_VERSION VERSION_LESS "5.5"))


# ----------------------------------------------------------------------------
# Apps compilation flags
# ----------------------------------------------------------------------------
if (COMPILER_CLANG)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-conversion")
endif()

# ----------------------------------------------------------------------------

# FIXME: src/libvideostitch-base/texture.hpp
if(COMPILER_GCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

include_directories(src)
if(WINDOWS)
  set(EXTERNAL_DEPS
    src/../../external_deps/)
  include_directories(${EXTERNAL_DEPS}/include)
  include_directories(${EXTERNAL_DEPS}/include/OculusSDK)
  include_directories(${EXTERNAL_DEPS}/include/openvr)
  find_library(OVR NAMES "LibOVR" PATHS ${EXTERNAL_DEPS}/lib/OculusSDK NO_DEFAULT_PATH)
  find_library(OPENVR NAMES "openvr_api" PATHS ${EXTERNAL_DEPS}/lib/openvr NO_DEFAULT_PATH)
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
  add_definitions(-DUNICODE)
endif(WINDOWS)

# ----------------------------------------------------------------------------
# Set common.hpp related variables
# ----------------------------------------------------------------------------
if(MSVC)
  set(COMMON_FLAGS "/FI\"libvideostitch-gui/common.hpp\"")
else()
  set(COMMON_FLAGS "-include libvideostitch-gui/common.hpp")
endif()
# ----------------------------------------------------------------------------

if(WINDOWS)
  set(VS_GUI "libvideostitch-gui")
  set(VS_BASE "libvideostitch-base")
else(WINDOWS)
  set(VS_GUI "videostitch-gui")
  set(VS_BASE "videostitch-base")
endif(WINDOWS)
add_subdirectory(src/libvideostitch-base)

if(BUILD_STUDIO OR BUILD_VAHANA)
  add_subdirectory(src/libvideostitch-gui)
endif()
if(BUILD_APPS_TESTS AND (BUILD_STUDIO OR BUILD_VAHANA))
  add_subdirectory(src/tests)
endif()

if(BUILD_STUDIO)
  add_subdirectory(src/batchstitcher)
  add_subdirectory(src/videostitch-studio-gui)
endif(BUILD_STUDIO)
if(BUILD_VAHANA)
  add_subdirectory(src/videostitch-live-gui)
endif(BUILD_VAHANA)
