if(DISABLE_JPEG)
  return()
endif()

set(SOURCE_FILES
    src/export.cpp
    src/jpgInput.cpp
    src/jpgOutput.cpp)

set(HEADER_FILES
    include/jpg.hpp
    include/jpgInput.hpp
    include/jpgOutput.hpp)

if(WINDOWS)
  set(PLUGIN_NAME jpg)
else()
  set(PLUGIN_NAME jpgPlugin)
endif()

vs_add_IO_library(${PLUGIN_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES} $<TARGET_OBJECTS:common>)

include_lib_vs_headers(${PLUGIN_NAME})
include_discovery_vs_headers(${PLUGIN_NAME})

target_include_directories(${PLUGIN_NAME} PRIVATE include)
target_include_directories(${PLUGIN_NAME} PRIVATE ../common/include)

if(LINUX)
  find_library(JPEG NAMES "jpeg")
elseif(APPLE)
  if(MACPORTS)
    find_library(JPEG NAMES jpeg PATHS /opt/local/lib)
  else()
    find_library(JPEG NAMES jpeg PATHS /usr/local/lib)
  endif()
else()
  find_package(JPEG)
endif()

if(APPLE)
  if(MACPORTS)
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE /opt/local/include)
  else()
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE /usr/local/include)
  endif()
endif()

if(APPLE OR LINUX)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${JPEG} ${VS_DISCOVERY})
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${JPEG_LIBRARIES} ${VS_DISCOVERY})
endif()
link_target_to_libvideostitch(${PLUGIN_NAME})
