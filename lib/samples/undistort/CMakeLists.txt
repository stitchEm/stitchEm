set(EXECUTABLE "undistort")
add_executable(${EXECUTABLE} "undistort.cpp" "../common/cmdUtils.cpp")
set_property(TARGET ${EXECUTABLE} PROPERTY FOLDER "lib/samples")

include_discovery_vs_headers(${EXECUTABLE})
include_lib_vs_headers(${EXECUTABLE})
link_target_to_libvideostitch(${EXECUTABLE})
target_link_libraries(${EXECUTABLE} PRIVATE ${VS_DISCOVERY})

