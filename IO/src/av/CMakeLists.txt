if(DISABLE_AV)
  return()
endif()

set(SOURCE_FILES
    src/avWriter.cpp
    src/baseAllocator.cpp
    src/d3dAllocator.cpp
    src/d3dDevice.cpp
    src/export.cpp
    src/libavReader.cpp
    src/netStreamReader.cpp
    src/timeoutUtil.cpp
    src/videoReader.cpp)
set(HEADER_FILES
    include/avWriter.hpp
    include/libavReader.hpp
    include/netStreamReader.hpp
    include/videoReader.hpp)

function(setup_av_plugin PLUGIN_NAME BACKEND_NAME USE_CUDA)
  # Set GPU core plugin output directories
  if(WINDOWS)
    set(VS_GPU_PLUGIN_DIR_NAME core_plugins_${BACKEND_NAME})

    # Set plugin output dir for the generic single-config case (e.g. make, ninja)
    set(VS_GPU_PLUGIN_DIR ${VS_OUT_DIR}/${CMAKE_BUILD_TYPE_LOW}/${VS_GPU_PLUGIN_DIR_NAME})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_GPU_PLUGIN_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_GPU_PLUGIN_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_GPU_PLUGIN_DIR})

    # Set plugin output dir for multi-config builds (e.g. MSVC, Xcode)
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
      string(TOLOWER ${OUTPUTCONFIG} OUTPUTCONFIG_LOW)
      set(VS_GPU_PLUGIN_DIR_${OUTPUTCONFIG_UP} ${VS_OUT_DIR}/${OUTPUTCONFIG_LOW}/${VS_GPU_PLUGIN_DIR_NAME})
      set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_GPU_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_GPU_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
      set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_GPU_PLUGIN_DIR_${OUTPUTCONFIG_UP}})
    endforeach()
  endif()

  vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:format_cuda_${USE_CUDA}>)
  include_lib_vs_headers(${PLUGIN_NAME})
  include_discovery_vs_headers(${PLUGIN_NAME})

  target_include_directories(${PLUGIN_NAME} PRIVATE include)
  target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
  target_include_directories(${PLUGIN_NAME} PRIVATE ../common/format/include)
  target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/Intel_Media_SDK)

  if(USE_CUDA)
    target_include_directories(${PLUGIN_NAME} PRIVATE ${CUDA_TOOLKIT_TARGET_DIR}/include)
    target_compile_definitions(${PLUGIN_NAME} PRIVATE SUP_NVENC SUP_NVDEC)
    find_package(CUDA REQUIRED)
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${CUDART})
  endif()

  set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

  set(FFMPEG_INCLUDE_PATH ${CMAKE_EXTERNAL_DEPS}/include/ffmpeg)
  if(APPLE_MACPORTS)
    set(FFMPEG_INCLUDE_PATH /opt/local/include)
  elseif(APPLE_BREW)
    set(FFMPEG_INCLUDE_PATH /usr/local/include)
  endif()

  target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE ${FFMPEG_INCLUDE_PATH})

  if(LINUX OR APPLE OR ANDROID)
    # VSA-5342: we're using functionality that has been deprecated in ffmpeg 3
    target_compile_options(${PLUGIN_NAME} PRIVATE -Wno-deprecated-declarations)
  endif()

  target_link_libraries(${PLUGIN_NAME} PRIVATE ${FFMPEG_libraries_cuda_${USE_CUDA}})
  if(WINDOWS)
    string(TOUPPER ${BACKEND_NAME} BACKEND_NAME_UP)
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_LIB_${BACKEND_NAME_UP}})
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${libmfxhw64} ${DirectX_LIB})
    set_property(TARGET ${PLUGIN_NAME} APPEND_STRING PROPERTY LINK_FLAGS "/NODEFAULTLIB:libcmt /NODEFAULTLIB:libcmtd")
  elseif(APPLE)
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_LIB_FAKE})
  else()
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_LIB_DEFAULT})
  endif()

  # Unit tests
  if(NOT WINDOWS)
    add_executable(AvUtilTest test/utilTest.cpp src/timeoutUtil.cpp)
    target_include_directories(AvUtilTest PRIVATE include)
    target_include_directories(AvUtilTest PRIVATE ../common/format/include)
    target_include_directories(AvUtilTest PRIVATE ${TESTING_INCLUDE})
    target_include_directories(AvUtilTest SYSTEM PRIVATE ${FFMPEG_INCLUDE_PATH})
    target_link_libraries(AvUtilTest PRIVATE ${VS_LIB_UNIT_TEST})
    set_property(TARGET AvUtilTest PROPERTY CXX_STANDARD 14)
    include_lib_vs_headers(AvUtilTest)
    include_discovery_vs_headers(AvUtilTest)
    add_test(NAME AvUtilTest COMMAND AvUtilTest)
  endif()
endfunction()


if(WINDOWS)
  if(GPU_BACKEND_CUDA)
    setup_av_plugin("av_cuda" "cuda" "ON")
  endif()
  if(GPU_BACKEND_OPENCL)
    setup_av_plugin("av_opencl" "opencl" "OFF")
  endif()
else()
  setup_av_plugin("avPlugin" "" ${GPU_BACKEND_CUDA})
endif()

# make I/O plugin list available to parent CMake project
set(VS_IO_LIBRARIES ${VS_IO_LIBRARIES} PARENT_SCOPE)
