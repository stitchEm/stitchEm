# Read this http://comments.gmane.org/gmane.comp.programming.tools.cmake.user/53930
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HID hid)

if(WIN_CHOCO)
  set(CERES_LIBS ${CONAN_LIBS})
  set(GLOG ${CONAN_LIBS})
else()
  find_debug_and_optimized_library(CERES_LIBS  "ceres"        "ceres-debug" "ceres"          "ceres")
  find_library(GLOG NAMES "libglog" PATHS "${CMAKE_EXTERNAL_LIB}/glog")
endif()


set(EIGEN3_INCLUDE_DIRS ${CMAKE_EXTERNAL_DEPS}/lib/eigen)

