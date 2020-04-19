# ----------------------------------------------------------------------------
# CUDA backend sources
# ----------------------------------------------------------------------------

set(CUDA_SOURCES
    src/backend/cuda/boundsKernel.cu
    src/backend/cuda/bilateral/bilateral.cu
    src/backend/cuda/core/panoRemapper.cu
    src/backend/cuda/core1/cutransform.cu
    src/backend/cuda/core1/mergerKernel.cu
    src/backend/cuda/core1/strip.cu
    src/backend/cuda/core1/voronoi.cu
    src/backend/cuda/coredepth/sphereSweep.cu
    src/backend/cuda/exampleKernel.cu
    src/backend/cuda/image/blur.cu
    src/backend/cuda/image/downsampler.cu
    src/backend/cuda/image/histogram.cu
    src/backend/cuda/image/imageOps.cu
    src/backend/cuda/image/imgExtract.cu
    src/backend/cuda/image/imgInsert.cu
    src/backend/cuda/image/reduce.cu
    src/backend/cuda/image/sampling.cu
    src/backend/cuda/image/rotate.cu
    src/backend/cuda/image/unpack.cu
    src/backend/cuda/input/checkerBoard.cu
    src/backend/cuda/input/maskInput.cu
    src/backend/cuda/mask/mergerMask.cu
    src/backend/cuda/mask/seamFinder.cu
    src/backend/cuda/memcpy.cu
    src/backend/cuda/processors/grid.cu
    src/backend/cuda/processors/photoCorr.cu
    src/backend/cuda/processors/tint.cu
    src/backend/cuda/render/geometry.cu
    src/backend/cuda/render/render.cu
    src/backend/cuda/score/scoringKernel.cu
    src/image/fill.cu
    src/image/transpose.cu
    src/output/anaglyphKernel.cu
    src/output/compositeOutputKernel.cu
    src/parallax/gpu/cuda/flowSequence.cu
    src/parallax/gpu/cuda/linearFlowWarper.cu
    src/parallax/gpu/cuda/mergerPair.cu
    src/parallax/gpu/cuda/simpleFlow.cu
    src/processors/maskoverlay.cu
    src/render/fillRenderer.cu
    src/util/imageProcessingGPUUtils.cu
    src/util/opticalFlowUtils.cu
    )

# message(STATUS "CUDA_SOURCES: ${CUDA_SOURCES}")

set(CUDA_BACKEND_SOURCES
    ${CUDA_BACKEND_SOURCES}
    src/backend/cuda/allocator.cpp
    src/backend/cuda/allocStats.cpp
    src/backend/cuda/buffer.cpp
    src/backend/cuda/context.cpp
    src/backend/cuda/device.cpp
    src/backend/cuda/deviceBuffer.cpp
    src/backend/cuda/deviceBuffer2D.cpp
    src/backend/cuda/deviceEvent.cpp
    src/backend/cuda/deviceHostBuffer.cpp
    src/backend/cuda/deviceStream.cpp
    src/backend/cuda/memcpy.cpp
    src/backend/cuda/opengl.cpp
    src/backend/cuda/surface.cpp
    src/maskinterpolation/inputMaskInterpolation.cpp
    src/cuda/error.cpp
    src/cuda/memory.cpp
    src/cuda/util.cpp
    src/image/histogramView.cpp
    src/mask/dijkstraShortestPath.cpp
    src/mask/mergerMask.cpp
    src/mask/mergerMaskAlgorithm.cpp
    src/mask/mergerMaskConfig.cpp
    src/mask/mergerMaskProgress.cpp
    src/mask/seamFinder.cpp
    src/parallax/flowSequence.cpp
    src/parallax/linearFlowWarper.cpp
    src/parallax/mergerPair.cpp
    src/parallax/simpleFlow.cpp
    src/parallax/spaceTransform.cpp
    src/stabilization/imuStabilization.cpp
    src/synchro/flashSync.cpp
    src/util/imageProcessingGPUUtils.cpp
    )

set(CUDA_BACKEND_HEADERS
    ${CUDA_BACKEND_HEADERS}
    src/backend/cuda/deviceBuffer.hpp
    src/backend/cuda/deviceEvent.hpp
    src/backend/cuda/deviceHostBuffer.hpp
    src/backend/cuda/deviceStream.hpp
    src/backend/cuda/image/colorArrayDevice.hpp
    src/maskinterpolation/inputMaskInterpolation.hpp
    src/cuda/error.hpp
    src/cuda/memory.hpp
    src/cuda/util.hpp
    src/gpu/buffer.hpp
    src/gpu/memcpy.hpp
    src/gpu/sharedBuffer.hpp
    src/gpu/uniqueBuffer.hpp
    src/image/histogramView.hpp
    include/libvideostitch/algorithm.hpp
    include/libvideostitch/context.hpp
    include/libvideostitch/gpu_device.hpp
    include/libvideostitch/matrix.hpp
    src/mask/dijkstraShortestPath.hpp
    src/mask/mergerMask.hpp
    src/mask/mergerMaskAlgorithm.hpp
    src/mask/mergerMaskConfig.hpp
    src/mask/mergerMaskProgress.hpp
    src/mask/seamFinder.hpp
    src/gpu/sharedBuffer.hpp
    src/gpu/uniqueBuffer.hpp
    src/parallax/flowSequence.hpp
    src/parallax/linearFlowWarper.hpp
    src/parallax/simpleFlow.hpp
    src/parallax/spaceTransform.hpp
    src/synchro/flashSync.hpp
    src/util/imageProcessingGPUUtils.hpp
    src/util/opticalFlowUtils.hpp
    )

# ----------------------------------------------------------------------------
# Create CUDA backend and libraries
# ----------------------------------------------------------------------------

# TODO: get rid of global CUDA header dependency
include_directories(${CUDA_INCLUDE_DIRS})

# TODO: remove
include_directories(src)

cuda_include_directories(src ${CMAKE_EXTERNAL_DEPS}/include)
cuda_include_directories(${VS_DISCOVERY_PUBLIC_HEADERS_DIR})
cuda_include_directories(${VS_LIB_PUBLIC_HEADERS_DIR})

if(MSVC)
  string(FIND ${CMAKE_CXX_FLAGS_RELEASE} "/GL" CONTAINS_GL_FLAG)
  # /GL option will cause CUDA runtime errors when building with Visual Studio 2017 and CUDA 10.2
  STRING(REPLACE " /GL" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# CMake object libs don't work with CUDA
cuda_compile(BACKEND_OBJECTS_CUDA ${CUDA_SOURCES})

if(MSVC)
  if (CONTAINS_GL_FLAG GREATER 0)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
  endif()
endif()

add_library(${VS_LIB_OBJECTS_CUDA} OBJECT ${CORE_LIB_SOURCES} ${CORE_LIB_HEADERS} ${CUDA_BACKEND_SOURCES} ${CUDA_BACKEND_HEADERS})
add_cppcheck(${VS_LIB_OBJECTS_CUDA} VS)

if(ANDROID)
  set(CUDA_PROPAGATE_HOST_FLAGS "FALSE")
  vs_lib_link_libraries("PUBLIC_CUDA" ${CUDA_LIBRARIES} ${NVTX})
  vs_lib_link_libraries("CUDA" ${GLEW} log)
  message(STATUS "CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
else()
  vs_lib_link_libraries("CUDA" ${CUDART} ${NVTX})
  vs_lib_link_libraries("CUDA" ${OpenGL} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLEW})
  vs_lib_link_libraries("PUBLIC_CUDA" ${NVML} ${CUDA_LIBRARIES})
  if(CMAKE_CROSSCOMPILING)
    # needed by ceres
    vs_lib_link_libraries("CUDA" -fopenmp)
  endif(CMAKE_CROSSCOMPILING)

  if(APPLE)
    vs_lib_link_libraries("CUDA" "-Wl,-F/Library/Frameworks -weak_framework CUDA")
  else()
    vs_lib_link_libraries("PUBLIC_CUDA" ${CUDA})
  endif()
endif()

