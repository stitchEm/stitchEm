set(CALIB "calibrationimport")
add_executable(${CALIB} "calibrationimport.cpp")
set_property(TARGET ${CALIB} PROPERTY FOLDER "lib/samples")

include_lib_vs_headers(${CALIB})
set_property(TARGET ${CALIB} PROPERTY CXX_STANDARD 14)
target_link_libraries(${CALIB} ${VS_LIB_DEFAULT})

