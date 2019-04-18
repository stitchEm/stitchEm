set(VS_LIB_SYSTEM_INCLUDES
    ${VS_LIB_SYSTEM_INCLUDES}
    ${VS_LIB_PUBLIC_HEADERS_DIR}/libvideostitch/MacOSX
    /opt/local/include
    )

find_library(IO_KIT IOKit REQUIRED)

if(APPLE_MACPORTS)
  find_package(Ceres REQUIRED)
  set(CERES_LIBS general ${CERES_LIBRARIES})

  set(EIGEN3_INCLUDE_DIRS /opt/local/include/eigen3)
else()
  find_library(CERES_OPTIMIZED_LIBRARY NAMES ceres ceres.1 PATHS ${CMAKE_EXTERNAL_DEPS}/lib/ceres)
  set(CERES_LIBS
      general ${CERES_OPTIMIZED_LIBRARY})
  set(EIGEN3_INCLUDE_DIRS ${CMAKE_EXTERNAL_DEPS}/lib/eigen)
endif()
