# safeguard against accidental misuse
if(NOT WINDOWS)
  message(FATAL_ERROR "Aja for Windows only!")
endif(NOT WINDOWS)

set(PLUGIN_NAME aja_64)

set(SOURCE_FILES
    export.cpp
    ntv2Discovery.cpp
    ntv2Helper.cpp
    ntv2plugin.cpp
    ntv2Reader.cpp
    ntv2Writer.cpp
    )

set(HEADER_FILES
    ntv2Discovery.hpp
    ntv2Helper.hpp
    ntv2plugin.hpp
    ntv2Reader.hpp
    ntv2Writer.hpp
    )

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)
target_compile_definitions(${PLUGIN_NAME} PRIVATE "MSWindows")
target_compile_definitions(${PLUGIN_NAME} PRIVATE "AJA_WINDOWS")
target_compile_definitions(${PLUGIN_NAME} PRIVATE "AJA_NO_AUTOIMPORT")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4005")
include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/aja/includes")
target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/aja/ajaapi")
target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/aja/classes")
target_include_directories(${PLUGIN_NAME} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/aja/winclasses")
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

find_debug_and_optimized_library(AJA_STUFFS  "aja/debug" "ajastuffdll_64d" "aja/release" "ajastuffdll_64")
find_debug_and_optimized_library(AJA_CLASSES "aja/debug" "classesDLL_64d"  "aja/release" "classesDLL_64")

target_link_libraries(${PLUGIN_NAME} PRIVATE ${AJA_STUFFS} ${AJA_CLASSES} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})
target_compile_definitions(${PLUGIN_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)

