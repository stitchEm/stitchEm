if(DISABLE_TIFF)
  return()
endif(DISABLE_TIFF)

set(SOURCE_FILES
    src/export.cpp
    src/tiffOutput.cpp)

set(HEADER_FILES
    include/tiffOutput.hpp)

if(WINDOWS)
  set(PLUGIN_NAME tiff)
else(WINDOWS)
  set(PLUGIN_NAME tiffPlugin)
endif(WINDOWS)

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

if(WINDOWS)
  find_library(TIFF NAMES "tiff")
elseif(LINUX)
  if(CMAKE_CROSSCOMPILING)
    target_include_directories(${PLUGIN_NAME} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include/libtiff)
    find_library(TIFF NAMES "tiff" PATHS ${CMAKE_EXTERNAL_DEPS}/lib NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  else()
      find_library(TIFF NAMES "tiff")
  endif()
elseif(APPLE)
    target_include_directories(${PLUGIN_NAME} PRIVATE "/opt/local/include")
    find_library(TIFF NAMES "tiff" PATHS "/opt/local/lib" REQUIRED)
endif()

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${TIFF} ${VS_DISCOVERY})
link_target_to_libvideostitch(${PLUGIN_NAME})