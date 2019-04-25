set(VS_LIB_SYSTEM_INCLUDES
    ${VS_LIB_SYSTEM_INCLUDES}
    ${VS_LIB_PUBLIC_HEADERS_DIR}/libvideostitch/MacOSX
    /opt/local/include
    )

find_library(IO_KIT IOKit REQUIRED)

find_package(Ceres REQUIRED)
set(CERES_LIBS general ${CERES_LIBRARIES})

if(MACPORTS)
  set(EIGEN3_INCLUDE_DIRS /opt/local/include/eigen3)
else()
  set(EIGEN3_INCLUDE_DIRS /usr/local/include/eigen3)
endif()
