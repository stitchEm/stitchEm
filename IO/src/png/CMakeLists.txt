set(SOURCE_FILES
    src/export.cpp
    src/Png.cpp
    src/pngInput.cpp
    src/pngOutput.cpp)

set(HEADER_FILES
    include/detail/Png.hpp
    include/pngInput.hpp
    include/pngOutput.hpp)

if(WINDOWS)
  set(PLUGIN_NAME png)
else(WINDOWS)
  set(PLUGIN_NAME pngPlugin)
endif(WINDOWS)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

if(APPLE)
    target_include_directories(${PLUGIN_NAME} PRIVATE "/opt/local/include")
else(APPLE)
    target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/libpng")
endif(APPLE)

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PNG_LIBRARY} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})

