if(DISABLE_RTMP)
  return()
endif(DISABLE_RTMP)

option(RTMP_QUICKSYNC "RTMP quicksync support" ${NANDROID})
option(RTMP_NVENC "RTMP NVidia Encode/Decode support" ON)
option(RTMP_NVENC_M "RTMP NVidia Encode/Decode support on Mobile Platform (TEGRA)" OFF)
option(USE_AVFORMAT "USE avformat to record RTMP" OFF)

if(APPLE OR TEGRA_DEMO OR LINUX)
  set(RTMP_QUICKSYNC OFF CACHE BOOL "" FORCE)
endif()

if(APPLE OR (NOT GPU_BACKEND_CUDA))
  set(RTMP_NVENC OFF CACHE BOOL "" FORCE)
elseif(TEGRA_DEMO)
  set(RTMP_NVENC OFF CACHE BOOL "" FORCE)
  set(RTMP_NVENC_M ON CACHE BOOL "" FORCE)
endif()

if(LINUX AND (NOT TEGRA_DEMO))
  set(USE_AVFORMAT ON CACHE BOOL "" FORCE)
endif()

set(SOURCE_FILES
    src/aacDecoder.cpp
    src/aacEncoder.cpp
    src/audioDecoder.cpp
    src/audioEncoder.cpp
    src/export.cpp
    src/lameDecoder.cpp
    src/lameEncoder.cpp
    src/metadataParser.cpp
    src/mockDecoder.cpp
    src/mockEncoder.cpp
    src/pcmDecoder.cpp
    src/rtmpClient.cpp
    src/rtmpPublisher.cpp
    src/videoDecoder.cpp
    src/videoEncoder.cpp
    src/x264Encoder.cpp
    src/rtmpDiscovery.cpp
    )


set(HEADER_FILES
    include/aacDecoder.hpp
    include/aacEncoder.hpp
    include/amfIncludes.hpp
    include/audioDecoder.hpp
    include/audioEncoder.hpp
    include/lameDecoder.hpp
    include/lameEncoder.hpp
    include/metadataParser.hpp
    include/mockEncoder.hpp
    include/pcmDecoder.hpp
    include/rtmpClient.hpp
    include/rtmpPublisher.hpp
    include/videoDecoder.hpp
    include/videoEncoder.hpp
    include/x264Encoder.hpp
    include/rtmpDiscovery.hpp
    )

if(RTMP_QUICKSYNC)
  set(SOURCE_FILES
      ${SOURCE_FILES}
      Intel/src/baseAllocator.cpp
      Intel/src/buffering.cpp
      Intel/src/d3dAllocator.cpp
      Intel/src/d3dDevice.cpp
      Intel/src/d3d11Allocator.cpp
      Intel/src/d3d11Device.cpp
      Intel/src/sysMemAllocator.cpp
      src/qsvDecoder.cpp
      src/qsvEncoder.cpp
      )

  set(HEADER_FILES
      ${HEADER_FILES}
      Intel/include/baseAllocator.hpp
      Intel/include/buffering.hpp
      Intel/include/d3dAllocator.hpp
      Intel/include/d3dDevice.hpp
      Intel/include/d3d11Allocator.hpp
      Intel/include/d3d11Device.hpp
      Intel/include/sysMemAllocator.hpp
      include/qsvEncoder.hpp
      )

  if(LINUX)
    set(SOURCE_FILES
        ${SOURCE_FILES}
        Intel/src/vaapiAllocator.cpp
        Intel/src/vaapiDevice.cpp
        )

    set(HEADER_FILES
        ${HEADER_FILES}
        Intel/include/vaapiAllocator.hpp
        Intel/include/vaapiDevice.hpp
        )
  endif(LINUX)
endif(RTMP_QUICKSYNC)


if(RTMP_NVENC)

  if(LINUX)
    include(FetchContent)
    FetchContent_Declare(
      nvenc_api
      GIT_REPOSITORY https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
      GIT_TAG n8.2.15.8
    )


    FetchContent_GetProperties(nvenc_api)
    if(NOT nvenc_api_POPULATED)
      FetchContent_Populate(nvenc_api)
      include_directories(${nvenc_api_SOURCE_DIR}/include/ffnvcodec/)
    endif()
  endif()

  set(SOURCE_FILES
      ${SOURCE_FILES}
      src/cuvid.cpp
      Nvidia/src/frameQueue.cpp
      src/nvenc.cpp
      )
  set(HEADER_FILES
      ${HEADER_FILES}
      Nvidia/include/frameQueue.hpp
      include/nvenc.hpp
      )
  find_package(CUDA REQUIRED)


elseif(RTMP_NVENC_M)
  set(SOURCE_FILES
      ${SOURCE_FILES}
      Nvidia/src/NvBuffer.cpp
      Nvidia/src/NvElement.cpp
      Nvidia/src/NvElementProfiler.cpp
      Nvidia/src/NvLogging.cpp
      Nvidia/src/NvV4l2Element.cpp
      Nvidia/src/NvV4l2ElementPlane.cpp
      Nvidia/src/NvVideoConverter.cpp
      Nvidia/src/NvVideoDecoder.cpp
      Nvidia/src/NvVideoEncoder.cpp
      src/NvV4l2Decoder.cpp
      src/NvV4l2Encoder.cpp
      )
  set(HEADER_FILES
      ${HEADER_FILES}
      Nvidia/include/NvBuffer.h
      Nvidia/include/NvElement.h
      Nvidia/include/NvElementProfiler.h
      Nvidia/include/NvLogging.h
      Nvidia/include/NvV4l2Element.h
      Nvidia/include/NvV4l2ElementPlane.h
      Nvidia/include/NvVideoConverter.h
      Nvidia/include/NvVideoDecoder.h
      Nvidia/include/NvVideoEncoder.h
      Nvidia/include/v4l2_nv_extensions.h
      include/NvV4l2Encoder.hpp
      )
endif()

if(WINDOWS)
  set(PLUGIN_NAME rtmp)
else(WINDOWS)
  set(PLUGIN_NAME rtmpPlugin)
endif(WINDOWS)

if(COMPILER_GCC)
  # FIXME: a lot of them in RTMP plugin
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
endif()

if(USE_AVFORMAT)
  vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:format_cuda_${GPU_BACKEND_CUDA}>)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE USE_AVFORMAT)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${FFMPEG_libraries_cuda_${GPU_BACKEND_CUDA}})
else(USE_AVFORMAT)
  vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)
endif(USE_AVFORMAT)

include_lib_vs_headers(${PLUGIN_NAME})

if(RTMP_QUICKSYNC)
  target_include_directories(${PLUGIN_NAME} PRIVATE Intel/include)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_QUICKSYNC)
endif()

if(RTMP_NVENC)
  target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/nvenc)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_NVENC)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_NVDEC)
  target_include_directories(${PLUGIN_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
  target_include_directories(${PLUGIN_NAME} PRIVATE Nvidia/include)
elseif(RTMP_NVENC_M)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_NVENC_M)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_NVDEC_M)
  target_include_directories(${PLUGIN_NAME} PRIVATE Nvidia/include)
  # needed to include libv4l2.h
  target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/v4l2)
  find_library(libv4l2 v4l2 HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${libv4l2})
endif(RTMP_NVENC)

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_include_directories(${PLUGIN_NAME} PRIVATE "${INTEL_MEDIA_SDK_PATH}\\include")
if(USE_AVFORMAT)
  target_include_directories(${PLUGIN_NAME} PRIVATE ../common/format/include)
endif(USE_AVFORMAT)
set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

if(WINDOWS)
  find_library(libx264 libx264 REQUIRED)
  find_library(librtmp librtmp REQUIRED)
  find_package(mp3lame CONFIG REQUIRED)
  set(DirectX_LIB
      dxva2
      d3d9
      dxgi
      d3d11
      Ws2_32
      Winmm)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${librtmp} ${libx264} mp3lame::mp3lame ${libmfxhw64} ${DirectX_LIB})
  set_property(TARGET ${PLUGIN_NAME} APPEND_STRING PROPERTY LINK_FLAGS "/NODEFAULTLIB:libcmt /NODEFAULTLIB:libcmtd")

  find_package(Git)
  include(ExternalProject)
  ############################################################
  ## Build FAAC
  ############################################################
  ExternalProject_Add(faac
    GIT_REPOSITORY https://github.com/stitchEm/faac.git
    GIT_TAG master
    PATCH_COMMAND ""
    CONFIGURE_COMMAND echo "Configure faac..."
    BUILD_COMMAND msbuild
        /p:Configuration=Release,Platform=x64
        /m <SOURCE_DIR>/libfaac/libfaac_dll.sln
    INSTALL_COMMAND echo "..."
    LICENSE_FILES COPYING
  )
  ExternalProject_Get_Property(faac SOURCE_DIR)

  target_include_directories(${PLUGIN_NAME} PRIVATE ${SOURCE_DIR}/include)
  set(Faac_LIBRARY ${SOURCE_DIR}/libfaac/ReleaseDLL/libfaac.lib)

  ############################################################
  ## Build FAAD2
  ############################################################
  ExternalProject_Add(faad2
    GIT_REPOSITORY https://github.com/stitchEm/faad2.git
    GIT_TAG xmms2
    PATCH_COMMAND ""
    CONFIGURE_COMMAND echo "Configure faad2..."
    BUILD_COMMAND msbuild
        /p:Configuration=Release,Platform=x64
        /m <SOURCE_DIR>/libfaad/libfaad2_dll.sln
    INSTALL_COMMAND echo "..."
    LICENSE_FILES COPYING
  )
  ExternalProject_Get_Property(faad2 SOURCE_DIR)
  add_dependencies(${PLUGIN_NAME} faad2 faac)

  set(libfaad ${SOURCE_DIR}/libfaad/ReleaseDLL/libfaad2.lib)
  target_include_directories(${PLUGIN_NAME} PRIVATE ${SOURCE_DIR}/include)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${Faac_LIBRARY} ${libfaad} ${CUVID})
endif()

if(LINUX OR ANDROID)
  if(LINUX)
    find_library(libx264 x264 REQUIRED)
    find_library(librtmp rtmp REQUIRED)
  else()
    find_library(librtmp rtmp HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(libx264 x264 HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  endif()
  if(CMAKE_CROSSCOMPILING)
    # needed to include x264.h
    target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/x264)
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/faac)
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/faad)
    find_library(libmp3lame mp3lame HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(libfaac faac HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(libvfaad faad HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  else()
    find_library(libmp3lame mp3lame)
    find_library(libfaac faac)
    find_library(libvfaad faad)
  endif()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${librtmp} ${libx264} ${libmp3lame} ${libfaac} ${libvfaad})
  if(RTMP_QUICKSYNC)
    find_library(igfx_s3dcontrol igfx_s3dcontrol PATH "${INTEL_MEDIA_SDK_PATH}\\igfx_s3dcontrol\\lib\\x64" REQUIRED)
    find_library(libva va HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_SYSTEM_ENVIRONMENT_PATH)
    find_library(libva_drm va-drm HINTS ${CMAKE_EXTERNAL_LIB} REQUIRED NO_SYSTEM_ENVIRONMENT_PATH)
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${libmfxhw64} ${libva} ${libva_drm} ${igfx_s3dcontrol})
  endif()
endif()

if(APPLE)
  find_library(librtmp rtmp REQUIRED)
  find_library(libx264 x264 REQUIRED)
  find_library(libfaac faac REQUIRED)
  find_library(libfaad faad.2 REQUIRED)
  find_library(libmp3lame mp3lame REQUIRED)
  if(MACPORTS)
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE /opt/local/include)
  endif()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${librtmp} ${libx264} ${libmp3lame} ${libfaac} ${libfaad})
endif()

link_target_to_libvideostitch(${PLUGIN_NAME})

if(RTMP_NVENC)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${CUDART} ${CUDA} ${CUVID})
endif()

if(WINDOWS)
  target_compile_definitions(${PLUGIN_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()


# Unit tests
if(NOT WINDOWS)
  add_executable(MetadataParsingTest test/orahMetadataParsingTest.cpp src/metadataParser.cpp)
  target_include_directories(MetadataParsingTest PRIVATE include)
  target_include_directories(MetadataParsingTest PRIVATE ${TESTING_INCLUDE})
  target_link_libraries(MetadataParsingTest PRIVATE ${VS_LIB_UNIT_TEST})
  set_property(TARGET MetadataParsingTest PROPERTY CXX_STANDARD 14)
  include_lib_vs_headers(MetadataParsingTest)
  add_test(NAME MetadataParsingTest COMMAND MetadataParsingTest)
endif()

