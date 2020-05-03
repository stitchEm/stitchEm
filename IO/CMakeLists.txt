project(IO)

# safeguard against accidental misuse
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

option(BUILD_IO_TESTS "Build plugins unit tests" ON)

# ----------------------------------------------------------------------------
# Helper macro to create a list of all I/O plugins
# ----------------------------------------------------------------------------
set(VS_IO_LIBRARIES)

macro(vs_add_IO_library lib_name)
  add_library(${ARGV})
  set(VS_IO_LIBRARIES
      ${VS_IO_LIBRARIES}
      ${lib_name}
      PARENT_SCOPE)
  set_property(TARGET ${lib_name} PROPERTY FOLDER "plugins")
  add_cppcheck(${lib_name} VS)
endmacro()


# ----------------------------------------------------------------------------
# Global plugin compilation flags
# ----------------------------------------------------------------------------
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB")
endif()


# ----------------------------------------------------------------------------
# Core plugin output directories
# ----------------------------------------------------------------------------

# Set plugin output dir for the generic single-config case (e.g. make, ninja)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_PLUGIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_PLUGIN_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_PLUGIN_DIR})

# Set plugin output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOLOWER ${OUTPUTCONFIG} OUTPUTCONFIG_LOW)
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_PLUGIN_DIR_${OUTPUTCONFIG_UP}})

  set(VS_PLUGIN_DIR ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_PLUGIN_DIR_NAME})
  set(VS_VAHANA_PLUGIN_DIR ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_VAHANA_PLUGIN_DIR_NAME})
  set(VS_TEST_DIR ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW})
  set(VS_STUDIO_DIR ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_STUDIO_DIR_NAME})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)


# ----------------------------------------------------------------------------
# Core plugins
# ----------------------------------------------------------------------------
option(DISABLE_AV   "Create AV   I/O    plugin" OFF)
option(DISABLE_BMP  "Create BMP  Input  plugin" ON)
option(DISABLE_JPEG "Create JPEG I/O    plugin" ${ANDROID})
option(DISABLE_TIFF "Create TIFF Output plugin" ${ANDROID})
option(DISABLE_MP4  "Create MP4  Input  plugin" ${NANDROID})

if(BUILD_IO_TESTS AND (${GPU_BACKEND_DEFAULT} STREQUAL CUDA OR NOT WINDOWS))
  add_subdirectory(src/test)
endif()

add_subdirectory(src/common)
add_subdirectory(src/av)
add_subdirectory(src/bmp)
add_subdirectory(src/jpg)
add_subdirectory(src/mp4)
add_subdirectory(src/pam)
add_subdirectory(src/png)
add_subdirectory(src/raw)
add_subdirectory(src/tiff)
add_subdirectory(src/exr)

# ----------------------------------------------------------------------------
# Vahana plugin output directories
# ----------------------------------------------------------------------------

# Set plugin output dir for the generic single-config case (e.g. make, ninja)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_VAHANA_PLUGIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_VAHANA_PLUGIN_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_VAHANA_PLUGIN_DIR})

# Set plugin output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)


# ----------------------------------------------------------------------------
# Vahana plugins
# ----------------------------------------------------------------------------
option(DISABLE_RTMP      "Create RTMP I/O      plugin" OFF)
option(DISABLE_PORTAUDIO "Create Portaudio I/O plugin" ${CMAKE_CROSSCOMPILING})

if(LINUX OR ANDROID)
  add_subdirectory(src/v4l2)
endif(LINUX OR ANDROID)

if(LINUX OR WINDOWS)
  add_subdirectory(src/portaudio)
endif(LINUX OR WINDOWS)

if(WINDOWS)
  find_library(libmfxhw64 libmfx_vs2015 HINTS "${INTEL_MEDIA_SDK_PATH}\\lib\\x64" REQUIRED)
  include_directories("${INTEL_MEDIA_SDK_PATH}\\include")
  include_directories("${INTEL_MEDIA_SDK_PATH}\\igfx_s3dcontrol\\include")
  #add_subdirectory(src/aja)
  add_subdirectory(src/decklink)
  add_subdirectory(src/magewell)
  add_subdirectory(src/magewellpro)
  add_subdirectory(src/ximea_2)
endif(WINDOWS)
add_subdirectory(src/rtmp)

# ----------------------------------------------------------------------------

# make I/O plugin list available to root CMake project
set(VS_IO_LIBRARIES ${VS_IO_LIBRARIES} PARENT_SCOPE)
