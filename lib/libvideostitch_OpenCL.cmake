option(DISABLE_OPENCL_SPIR "Skip OpenCL offline compilation and ship kernel sources")
option(CL_ARGS_WORKAROUND "Redefining functions to support OpenCL limitation in parameters format")

if(NOT DISABLE_OPENCL_SPIR)
  if (LINUX OR ANDROID)
    message("No Alternative OpenCL SPIR compiler for Linux")
  else()
    option(ALTERNATIVE_OPENCL_SPIR "GENERATE an alternative SPIR that will be used if something goes wrong with the main SPIR" ON)
  endif()
endif()

set(CL_BACKEND_SOURCES
    src/backend/cl/bilateral/bilateral.cpp
    src/backend/cl/allocator.cpp
    src/backend/cl/allocStats.cpp
    src/backend/cl/deviceBuffer.cpp
    src/backend/cl/deviceBuffer2D.cpp
    src/backend/cl/deviceHostBuffer.cpp
    src/backend/cl/context.cpp
    src/backend/cl/binaryCache.cpp
    src/backend/cl/cl_error.cpp
    src/backend/cl/context.cpp
    src/backend/cl/core1/boundsKernel.cpp
    src/backend/cl/core1/mergerKernel.cpp
    src/backend/cl/core1/panoRemapper.cpp
    src/backend/cl/core1/strip.cpp
    src/backend/cl/core1/transform.cpp
    src/backend/cl/core1/voronoiKernel.cpp
    src/backend/cl/coredepth/sweep.cpp
    src/backend/cl/device.cpp
    src/backend/cl/deviceBuffer.cpp
    src/backend/cl/deviceEvent.cpp
    src/backend/cl/deviceHostBuffer.cpp
    src/backend/cl/deviceStream.cpp
    src/backend/cl/exampleKernel.cpp
    src/backend/cl/gpuContext.cpp
    src/backend/cl/image/blur.cpp
    src/backend/cl/image/cl_downsampler.cpp
    src/backend/cl/image/imageOps.cpp
    src/backend/cl/image/imgExtract.cpp
    src/backend/cl/image/imgInsert.cpp
    src/backend/cl/image/reduce.cpp
    src/backend/cl/image/sampling.cpp
    src/backend/cl/image/rotate.cpp
    src/backend/cl/image/unpack.cpp
    src/backend/cl/input/checkerBoard.cpp
    src/backend/cl/input/maskInput.cpp
    src/backend/cl/kernel.cpp
    src/backend/cl/memcpy.cpp
    src/backend/cl/opengl.cpp
    src/backend/cl/processors/grid.cpp
    src/backend/cl/processors/maskoverlay.cpp
    src/backend/cl/processors/photoCorr.cpp
    src/backend/cl/processors/tint.cpp
    src/backend/cl/score/scoringKernel.cpp
    src/backend/cl/render/geometry.cpp
    src/backend/cl/render/render.cpp
    src/backend/cl/surface.cpp
    src/backend/cl/deviceStream.cpp
    src/backend/cl/vectorTypes.cpp
    src/mask/mergerMaskAlgorithm_opencl.cpp
    src/mask/mergerMaskConfig.cpp
    src/synchro/flashSync_opencl.cpp
    )

if(NOT ANDROID)
  find_package(OpenCL REQUIRED)
  if(WINDOWS)
    find_library(GLEW NAMES glew32s PATHS ${CMAKE_EXTERNAL_DEPS}/lib/GL NO_DEFAULT_PATH)
  else()
    find_package(GLEW REQUIRED)
  endif()
  find_package(OpenGL REQUIRED)
endif()

message(STATUS "--- OpenCL GPU backend ---")
message(STATUS "OpenCL version: ${OpenCL_VERSION_STRING}")
message(STATUS "OpenCL includes: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
message(STATUS "")

set(OPENCL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/spir")

add_subdirectory(src/backend/cl)

add_library(${VS_LIB_OBJECTS_OPENCL} OBJECT ${CORE_LIB_SOURCES} ${CL_BACKEND_SOURCES})
add_cppcheck(${VS_LIB_OBJECTS_OPENCL} VS)

# ----------------------------------------------------------------------------
# Add compile definitions
# ----------------------------------------------------------------------------
if(DISABLE_OPENCL_SPIR)
  target_compile_definitions(${VS_LIB_OBJECTS_OPENCL} PRIVATE "DISABLE_OPENCL_SPIR")
endif()
if(CL_ARGS_WORKAROUND)
  message(WARNING "Careful: OpenCL kernel parameters and function redefined to handle some online compilation limitation!")
  target_compile_definitions(${VS_LIB_OBJECTS_OPENCL} PRIVATE "CL_ARGS_WORKAROUND")
endif()
# ----------------------------------------------------------------------------

add_dependencies(${VS_LIB_OBJECTS_OPENCL} opencl_compilation)

target_include_directories(${VS_LIB_OBJECTS_OPENCL} PRIVATE SYSTEM ${OpenCL_INCLUDE_DIRS})
target_include_directories(${VS_LIB_OBJECTS_OPENCL} PRIVATE ${OPENCL_BINARY_DIR})

if(DISABLE_OPENCL_SPIR)
  target_compile_definitions(${VS_LIB_OBJECTS_OPENCL} PRIVATE DISABLE_OPENCL_SPIR)
else(DISABLE_OPENCL_SPIR)
  if (ALTERNATIVE_OPENCL_SPIR)
    target_compile_definitions(${VS_LIB_OBJECTS_OPENCL}  PRIVATE ALTERNATIVE_OPENCL_SPIR)
    add_definitions(-DALTERNATIVE_OPENCL_SPIR)
  endif(ALTERNATIVE_OPENCL_SPIR)
endif(DISABLE_OPENCL_SPIR)

# different names on different implementations of find_package(OpenCL)
vs_lib_link_libraries("PUBLIC_OPENCL" ${OpenCL_LIBRARIES})
vs_lib_link_libraries("OPENCL" ${OpenGL_LIBRARIES} ${OPENGL_LIBRARIES} GLEW::GLEW)

set(BACKEND_OBJECTS_OPENCL )
