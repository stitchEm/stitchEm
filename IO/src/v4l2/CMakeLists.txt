set(PLUGIN_NAME v4l2Plugin)

set(SOURCE_FILES
    src/export.cpp
    src/v4l2.cpp)

set(HEADER_FILES
    include/v4l2.hpp)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})

