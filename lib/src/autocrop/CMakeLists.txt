find_package(OpenCV)

set(AUTOCROP_SOURCES
    autoCrop.cpp
    autoCropConfig.cpp
    autoCropAlgorithm.cpp)

set(AUTOCROP_HEADERS
    autoCrop.hpp
    autoCropConfig.hpp
    autoCropAlgorithm.hpp)

set(VS_LIB_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)


function(configure_target BACKEND_NAME)
  add_library(autocrop_${BACKEND_NAME} OBJECT ${AUTOCROP_SOURCES} ${AUTOCROP_HEADERS})
  add_cppcheck(autocrop_${BACKEND_NAME} VS)
  set_property(TARGET autocrop_${BACKEND_NAME} PROPERTY FOLDER "lib")

  target_include_directories(autocrop_${BACKEND_NAME}
                             SYSTEM
                             PRIVATE ${CMAKE_EXTERNAL_DEPS}/include
                             PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/ceres
                             PRIVATE ${EIGEN3_INCLUDE_DIRS}
                             PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/glog
                             PRIVATE ${OpenCV_INCLUDE_DIRS})

  if (WINDOWS)
    target_compile_definitions(autocrop_${BACKEND_NAME} PRIVATE VS_LIB_COMPILATION NOMINMAX _USE_MATH_DEFINES)
  endif(WINDOWS)

  include_discovery_vs_headers(autocrop_${BACKEND_NAME})
  include_lib_vs_headers(autocrop_${BACKEND_NAME})

  if (UNIX)
    set_property(TARGET autocrop_${BACKEND_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endif (UNIX)

  target_include_directories(autocrop_${BACKEND_NAME} PRIVATE ${VS_LIB_ROOT_SRC_DIR})
  target_compile_definitions(autocrop_${BACKEND_NAME} PRIVATE VS_${BACKEND_NAME})

  foreach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
    target_include_directories(autocrop_${BACKEND_NAME} SYSTEM PRIVATE ${vs_lib_sys_include})
  endforeach()

  set_property(TARGET autocrop_${BACKEND_NAME} PROPERTY CXX_STANDARD 14)
endfunction()

if(GPU_BACKEND_CUDA)
  configure_target("CUDA")
endif()
if(GPU_BACKEND_OPENCL)
  configure_target("OPENCL")
endif()

