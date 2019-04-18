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

target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/ximea")
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

find_library(XIMEA_LIB NAMES "xiapi64" PATHS ${CMAKE_EXTERNAL_DEPS}/lib/Ximea/release/ REQUIRED NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY} ${XIMEA_LIB})
link_target_to_libvideostitch(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)
