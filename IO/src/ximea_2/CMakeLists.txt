# safeguard against accidental misuse
if(NOT WINDOWS)
  message(FATAL_ERROR "Ximea for Windows only!")
endif(NOT WINDOWS)

set(PLUGIN_NAME ximea_64)

set(SOURCE_FILES
    export.cpp
    ximeaDiscovery.cpp
    ximeaReader.cpp
    )

set(HEADER_FILES
    ximeaDiscovery.hpp
    ximeaReader.hpp
    )

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE ${XIMEA_PATH})
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

find_library(XIMEA_LIB NAMES "xiapi64" HINTS "${XIMEA_PATH}\\x64" REQUIRED)

target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY} ${XIMEA_LIB})
link_target_to_libvideostitch(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)
