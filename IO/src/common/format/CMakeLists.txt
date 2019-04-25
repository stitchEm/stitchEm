if(DISABLE_AV)
  return()
endif(DISABLE_AV)

cmake_minimum_required(VERSION 3.2)

set(COMMON_SOURCES
    src/avMuxer.cpp
    src/filemuxer.cpp
    src/livemuxer.cpp
    src/muxer.cpp
    src/qt-faststart.cpp
    src/util.cpp
    )

set(COMMON_HEADERS
    include/avMuxer.hpp
    include/filemuxer.hpp
    include/livemuxer.hpp
    include/muxer.hpp
    include/qt-faststart.hpp
    include/util.hpp
    )

function(setup_format BACKEND_NAME USE_CUDA)
  set(LIST_NAME format_cuda_${USE_CUDA})
  add_library(${LIST_NAME} OBJECT ${COMMON_SOURCES} ${COMMON_HEADERS})
  set_property(TARGET ${LIST_NAME} PROPERTY FOLDER "plugins")
  set_property(TARGET ${LIST_NAME} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${LIST_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  add_cppcheck(${LIST_NAME} VS)

  include_lib_vs_headers(${LIST_NAME})
  target_include_directories(${LIST_NAME} PRIVATE include)
  target_include_directories(${LIST_NAME} PRIVATE ../include)
  set(FFMPEG_INCLUDE_PATH ${CMAKE_EXTERNAL_DEPS}/include/ffmpeg)

  target_include_directories(${LIST_NAME} PRIVATE ${FFMPEG_INCLUDE_PATH})
  if(LINUX OR APPLE)
    # VSA-5342: we're using functionality that has been deprecated in ffmpeg 3
    target_compile_options(${LIST_NAME} PRIVATE -Wno-deprecated-declarations)
  endif(LINUX OR APPLE)

  if(LINUX)
      find_library(FFMPEG_avcodec avcodec REQUIRED)
      find_library(FFMPEG_avformat avformat REQUIRED)
      find_library(FFMPEG_avutil avutil REQUIRED)
  elseif(ANDROID)
      find_library(FFMPEG_avcodec avcodec PATHS ${CMAKE_EXTERNAL_DEPS}/lib REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
      find_library(FFMPEG_avformat avformat PATHS ${CMAKE_EXTERNAL_DEPS}/lib REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
      find_library(FFMPEG_avutil avutil PATHS ${CMAKE_EXTERNAL_DEPS}/lib REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  endif()

  if(WINDOWS)
    set(FFMPEG_LIB_PATH ${CMAKE_EXTERNAL_DEPS}/lib/ffmpeg)
    find_library(FFMPEG_avcodec NAMES avcodec PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
    find_library(FFMPEG_avformat NAMES avformat PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
    find_library(FFMPEG_avutil NAMES avutil PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
  endif(WINDOWS)

  if(APPLE)
    if(MACPORTS)
      find_library(FFMPEG_avcodec NAMES avcodec)
      find_library(FFMPEG_avformat NAMES avformat)
      find_library(FFMPEG_avutil NAMES avutil)
      target_include_directories(${LIST_NAME} PRIVATE /opt/local/include)
    else()
      find_library(FFMPEG_avcodec NAMES avcodec)
      find_library(FFMPEG_avformat NAMES avformat)
      find_library(FFMPEG_avutil NAMES avutil)
      target_include_directories(${LIST_NAME} PRIVATE /usr/local/include)
    endif()
  endif()
  set(FFMPEG_libraries_cuda_${USE_CUDA} ${FFMPEG_avcodec} ${FFMPEG_avutil} ${FFMPEG_avformat} CACHE STRING "FFMPEG LIBRARIES")

endfunction()

if(GPU_BACKEND_CUDA)
  setup_format("cuda" "ON")
endif()
if(GPU_BACKEND_OPENCL)
  setup_format("opencl" "OFF")
endif()

