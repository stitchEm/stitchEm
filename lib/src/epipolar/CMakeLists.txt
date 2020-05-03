find_package(OpenCV)

set(EPIPOLAR_SOURCES
    epipolarCurvesConfig.cpp
    epipolarCurvesAlgorithm.cpp)

set(EPIPOLAR_HEADERS
    epipolarCurvesConfig.hpp
    epipolarCurvesAlgorithm.hpp)

set(VS_LIB_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)


function(configure_target BACKEND_NAME)
  add_library(epipolar_${BACKEND_NAME} OBJECT ${EPIPOLAR_SOURCES} ${EPIPOLAR_HEADERS})
  add_cppcheck(epipolar_${BACKEND_NAME} VS)
  set_property(TARGET epipolar_${BACKEND_NAME} PROPERTY FOLDER "lib")

  target_include_directories(epipolar_${BACKEND_NAME}
                             SYSTEM
                             PRIVATE ${EIGEN3_INCLUDE_DIRS}
                             PRIVATE ${OpenCV_INCLUDE_DIRS})

  if (WINDOWS)
    target_compile_definitions(epipolar_${BACKEND_NAME} PRIVATE VS_LIB_COMPILATION NOMINMAX _USE_MATH_DEFINES)
  endif(WINDOWS)

  if (UNIX)
    set_property(TARGET epipolar_${BACKEND_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endif (UNIX)

  target_include_directories(epipolar_${BACKEND_NAME} PRIVATE ${VS_LIB_ROOT_SRC_DIR})
  target_compile_definitions(epipolar_${BACKEND_NAME} PRIVATE VS_${BACKEND_NAME})

  include_lib_vs_headers(epipolar_${BACKEND_NAME})
  include_discovery_vs_headers(epipolar_${BACKEND_NAME})

  foreach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
    target_include_directories(epipolar_${BACKEND_NAME} SYSTEM PRIVATE ${vs_lib_sys_include})
  endforeach()

  set_property(TARGET epipolar_${BACKEND_NAME} PROPERTY CXX_STANDARD 14)
endfunction()

if(GPU_BACKEND_CUDA)
  configure_target("CUDA")
endif()
if(GPU_BACKEND_OPENCL)
  configure_target("OPENCL")
endif()

