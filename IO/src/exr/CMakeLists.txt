if(DISABLE_EXR)
  return()
endif(DISABLE_EXR)

find_package(OpenEXR REQUIRED)

set(SOURCE_FILES
    src/export.cpp
    src/exrInput.cpp
    src/exrOutput.cpp)

set(HEADER_FILES
    include/exrInput.hpp
    include/exrOutput.hpp)

if(WINDOWS)
  set(PLUGIN_NAME exr)
else(WINDOWS)
  set(PLUGIN_NAME exrPlugin)
endif(WINDOWS)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE ${OpenEXR_INCLUDE_DIRS})

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${OpenEXR_LIBRARIES} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})

