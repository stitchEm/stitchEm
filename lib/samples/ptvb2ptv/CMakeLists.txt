set(PTVB2PTV "ptvb2ptv")
add_executable(${PTVB2PTV} "ptvb2ptv.cpp")
include_lib_vs_headers(${PTVB2PTV})
set_property(TARGET ${PTVB2PTV} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PTVB2PTV} ${VS_LIB_DEFAULT})
set_property(TARGET ${PTVB2PTV} PROPERTY FOLDER "lib/samples")

