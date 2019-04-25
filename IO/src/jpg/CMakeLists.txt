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

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 14)

if(LINUX)
  find_library(JPEG NAMES "jpeg")
elseif(APPLE)
  if(MACPORTS)
    find_library(JPEG NAMES jpeg PATHS /opt/local/lib)
  else()
    find_library(JPEG NAMES jpeg PATHS /usr/local/lib)
  endif()
else()
  find_library(TURBO_JPEG NAMES turbojpeg turbojpeg.0 PATHS "${CMAKE_EXTERNAL_DEPS}/lib/libjpeg-turbo" NO_DEFAULT_PATH)
  find_library(JPEG NAMES jpeg jpeg.62 PATHS "${CMAKE_EXTERNAL_DEPS}/lib/libjpeg-turbo" NO_DEFAULT_PATH)
endif()

if(APPLE)
  if(MACPORTS)
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE /opt/local/include)
  else()
    target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE /usr/local/include)
  endif()
elseif(WINDOWS)
  target_include_directories(${PLUGIN_NAME} SYSTEM PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/libjpeg-turbo")
endif()

if(APPLE OR LINUX)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${JPEG} ${VS_DISCOVERY})
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${TURBO_JPEG} ${JPEG} ${VS_DISCOVERY})
endif()
link_target_to_libvideostitch(${PLUGIN_NAME})
