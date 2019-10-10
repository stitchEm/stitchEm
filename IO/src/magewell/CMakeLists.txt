# safeguard against accidental misuse
if(NOT WINDOWS)
  message(FATAL_ERROR "magewell for Windows only!")
endif(NOT WINDOWS)

set(PLUGIN_NAME magewell)

set(SOURCE_FILES
    export.cpp
    magewell_discovery.cpp
    magewell_helpers.cpp
    magewell_reader.cpp
    )

set(HEADER_FILES
    magewell_discovery.hpp
    magewell_helpers.hpp
    magewell_reader.hpp
    )

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)
include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_include_directories(${PLUGIN_NAME} PRIVATE "${MAGEWELL_PATH}\\SDKv2\\Include")

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

find_library(MAGEWELL_STREAM2 "LibXIStream2" HINTS "${MAGEWELL_PATH}\\SDKv2\\Lib\\x64\\Release")
find_library(MAGEWELL_PROPERTY "LibXIProperty" HINTS "${MAGEWELL_PATH}\\SDKv2\\Lib\\x64\\Release")
target_link_libraries(${PLUGIN_NAME} PRIVATE ${VS_DISCOVERY} ${MAGEWELL_STREAM2} ${MAGEWELL_PROPERTY})
link_target_to_libvideostitch(${PLUGIN_NAME})
target_compile_definitions(${PLUGIN_NAME} PRIVATE _UNICODE UNICODE)