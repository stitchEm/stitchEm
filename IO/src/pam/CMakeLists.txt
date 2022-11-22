set(SOURCE_FILES
    src/export.cpp
    src/pamOutput.cpp
    src/pgmOutput.cpp
    src/pnm.cpp
    src/ppmOutput.cpp
    src/yuv420pOutput.cpp)

set(HEADER_FILES
    include/pamOutput.hpp
    include/pgmOutput.hpp
    include/pnm.hpp
    include/ppmOutput.hpp
    include/yuv420pOutput.hpp)

if(WINDOWS)
  set(PLUGIN_NAME pam)
else(WINDOWS)
  set(PLUGIN_NAME pamPlugin)
endif(WINDOWS)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

link_target_to_libvideostitch(${PLUGIN_NAME})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY})

