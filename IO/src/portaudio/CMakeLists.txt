if(DISABLE_PORTAUDIO)
  return()
endif(DISABLE_PORTAUDIO)

set(SOURCE_FILES
  src/export.cpp
  src/portAudioDiscovery.cpp
  src/portAudioReader.cpp
  src/portAudioWriter.cpp)
set(HEADER_FILES
  include/portAudioDiscovery.hpp
  include/portAudioReader.hpp
  include/portAudioWriter.hpp)

if(WINDOWS)
  set(PLUGIN_NAME portaudio)
else()
  set(PLUGIN_NAME portaudioPlugin)
endif()

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)
target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/portaudio)
set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

if(WINDOWS)
  find_library(PORTAUDIO NAMES "portaudio_x64" PATHS ${CMAKE_EXTERNAL_DEPS}/lib/portaudio REQUIRED)
else()
  find_library(PORTAUDIO NAMES "portaudio" REQUIRED)
endif()

# FIXME: portAudioReader.cpp
if(COMPILER_GCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

target_link_libraries(${PLUGIN_NAME} PRIVATE ${PORTAUDIO} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})

