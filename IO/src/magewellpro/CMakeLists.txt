# safeguard against accidental misuse
if(NOT WINDOWS)
  message(FATAL_ERROR "magewellpro for Windows only!")
endif(NOT WINDOWS)

set(PLUGIN_NAME magewellpro)

set(SOURCE_FILES
    export.cpp
    magewellDiscovery.cpp
    magewell_helpers.cpp
    magewellReader.cpp
    )

set(HEADER_FILES
    magewellDiscovery.hpp
    magewell_helpers.hpp
    magewellReader.hpp
    )
vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)
include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_include_directories(${PLUGIN_NAME} PRIVATE "${MAGEWELL_PATH}\\SDKv3\\Include")

find_library(MAGEWELL_DEVICE "LibMWCapture" HINTS "${MAGEWELL_PATH}\\SDKv3\\Lib\\x64")
target_link_libraries(${PLUGIN_NAME} PRIVATE ${MAGEWELL_DEVICE} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})
target_compile_definitions(${PLUGIN_NAME} PRIVATE _UNICODE UNICODE)