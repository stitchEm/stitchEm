find_package(OpenCV)

set(CALIBRATION_SOURCES
    boundedValue.cpp
    camera.cpp
    camera_fisheye.cpp
    camera_nextfisheye.cpp
    camera_perspective.cpp
    calibration.cpp
    calibrationAlgorithm.cpp
    calibrationAlgorithmBase.cpp
    calibrationConfig.cpp
    calibrationOnlineAlgorithm.cpp
    calibrationProgress.cpp
    calibrationRefinement.cpp
    calibrationPresetsMakerConfig.cpp
    calibrationPresetsMakerAlgorithm.cpp
    calibrationDeshuffling.cpp
    calibrationSyntheticKeypoints.cpp
    controlPointFilter.cpp
    eigengeometry.cpp
    jacobians.cpp
    keypointExtractor.cpp
    keypointMatcher.cpp
    rigBuilder.cpp
    rigGraph.cpp
    calibrationUtils.cpp)

set(CALIBRATION_HEADERS
    boundedParameterization.hpp
    boundedValue.hpp
    calibration.hpp
    calibrationAlgorithm.hpp
    calibrationAlgorithmBase.hpp
    calibrationConfig.hpp
    calibrationOnlineAlgorithm.hpp
    calibrationPresetsMakerAlgorithm.hpp
    calibrationPresetsMakerConfig.hpp
    calibrationProgress.hpp
    calibrationRefinement.hpp
    calibrationUtils.hpp
    camera.hpp
    camera_fisheye.hpp
    camera_nextfisheye.hpp
    camera_perspective.hpp
    controlPointFilter.hpp
    cvImage.hpp
    eigengeometry.hpp
    inputDistance.hpp
    jacobians.hpp
    keypointExtractor.hpp
    keypointMatcher.hpp
    ransacRotationSolver.hpp
    rigBuilder.hpp
    rigGraph.hpp
    rotationDistance.hpp
    rotationEstimation.hpp
    so3Parameterization.hpp)

set(VS_LIB_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)


function(configure_target BACKEND_NAME)
  add_library(calibration_${BACKEND_NAME} OBJECT ${CALIBRATION_SOURCES} ${CALIBRATION_HEADERS})
  add_cppcheck(calibration_${BACKEND_NAME} VS)
  set_property(TARGET calibration_${BACKEND_NAME} PROPERTY FOLDER "lib")

  if (WINDOWS)
    target_compile_definitions(calibration_${BACKEND_NAME} PRIVATE VS_LIB_COMPILATION NOMINMAX _USE_MATH_DEFINES)
  endif(WINDOWS)

  if (UNIX)
    set_property(TARGET calibration_${BACKEND_NAME} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endif (UNIX)

  target_include_directories(calibration_${BACKEND_NAME} PRIVATE ${VS_LIB_ROOT_SRC_DIR})
  target_compile_definitions(calibration_${BACKEND_NAME} PRIVATE VS_${BACKEND_NAME})

  foreach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
    target_include_directories(calibration_${BACKEND_NAME} SYSTEM PRIVATE ${vs_lib_sys_include})
  endforeach()

  include_lib_vs_headers(calibration_${BACKEND_NAME})
  include_discovery_vs_headers(calibration_${BACKEND_NAME})

  target_include_directories(calibration_${BACKEND_NAME}
                             SYSTEM
                             PRIVATE ${EIGEN3_INCLUDE_DIRS}
                             PRIVATE ${OpenCV_INCLUDE_DIRS})

endfunction()

if(GPU_BACKEND_CUDA)
  configure_target("CUDA")
endif()
if(GPU_BACKEND_OPENCL)
  configure_target("OPENCL")
endif()

