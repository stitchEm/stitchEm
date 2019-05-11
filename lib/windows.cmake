# Read this http://comments.gmane.org/gmane.comp.programming.tools.cmake.user/53930
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HID hid)

if(WIN_CHOCO)
  set(CERES_LIBS CONAN_LIBS_TODO_FIND_CERES)
  set(GLOG CONAN_LIBS_TODO_GLOG_CERES_DEP)
else()
  find_debug_and_optimized_library(CERES_LIBS  "ceres"        "ceres-debug" "ceres"          "ceres")
  find_library(GLOG NAMES "libglog" PATHS "${CMAKE_EXTERNAL_LIB}/glog")
endif()


if(WIN_CHOCO)
  set(EIGEN3_INCLUDE_DIRS ${CONAN_INCLUDE_DIRS_EIGEN})
else()
  set(EIGEN3_INCLUDE_DIRS ${CMAKE_EXTERNAL_DEPS}/lib/eigen)
endif()

if(WIN_CHOCO)
  set(VS_LIB_SYSTEM_INCLUDES
      ${VS_LIB_SYSTEM_INCLUDES}
      ${CONAN_INCLUDE_DIRS}
      )
endif()
