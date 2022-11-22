if(WINDOWS)
  set(PLUGIN_NAME raw)
else(WINDOWS)
  set(PLUGIN_NAME rawPlugin)
endif(WINDOWS)

set(SOURCE_FILES
    src/export.cpp
    src/rawOutput.cpp)

set(HEADER_FILES
    include/rawOutput.hpp)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})

