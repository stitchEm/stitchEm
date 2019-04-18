if(DISABLE_MP4)
  return()
endif(DISABLE_MP4)

set(PLUGIN_NAME mp4Plugin)

set(SOURCE_FILES
    mp4Input.cpp
    export.cpp)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} $<TARGET_OBJECTS:common>)
include_lib_vs_headers(${PLUGIN_NAME})
target_include_directories(${PLUGIN_NAME} PRIVATE .)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PLUGIN_NAME} PRIVATE mediandk ${OPENMAXAL_LIB} log)
link_target_to_libvideostitch(${PLUGIN_NAME})

