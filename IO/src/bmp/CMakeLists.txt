if(DISABLE_BMP)
  return()
endif(DISABLE_BMP)

set(SOURCE_FILES
  bmpInput.cpp
  export.cpp)

if(WINDOWS)
    set(PLUGIN_NAME bmp)
else(WINDOWS)
    set(PLUGIN_NAME bmpPlugin)
endif(WINDOWS)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} $<TARGET_OBJECTS:common>)
include_lib_vs_headers(${PLUGIN_NAME})
target_include_directories(${PLUGIN_NAME} PRIVATE .)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

link_target_to_libvideostitch(${PLUGIN_NAME})

