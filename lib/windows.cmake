# Read this http://comments.gmane.org/gmane.comp.programming.tools.cmake.user/53930
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HID hid)

find_library(CERES_LIBS "ceres")
find_library(GLOG NAMES "glog")