# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

find_package(OpenGL REQUIRED)

set(VIDEOSTITCH_BASE_SOURCES
    calibrationImport.cpp
    file.cpp
    geometry.cpp
    interactiverenderer.cpp
    interactivewidget.cpp
    logmanager.cpp
    msgboxhandler.cpp
    projection.cpp
    renderer.cpp
    texture.cpp
    videowidget.cpp
    vslog.cpp
    yprsignalcaps.cpp
    )

set(VIDEOSTITCH_BASE_HEADERS
    appslogging.hpp
    calibrationImport.hpp
    common-config.hpp
    file.hpp
    frame.hpp
    geometry.hpp
    lambdathread.hpp
    lockingproxy.hpp
    logmanager.hpp
    interactiverenderer.hpp
    interactivewidget.hpp
    linkhelpers.hpp
    msgboxhandler.hpp
    projection.hpp
    renderer.hpp
    singleton.hpp
    texture.hpp
    videowidget.hpp
    vslog.hpp
    yprsignalcaps.hpp
    )

if (WINDOWS)
  set(VIDEOSTITCH_BASE_SOURCES
      ${VIDEOSTITCH_BASE_SOURCES}
      matrices.cpp
      mirrorwidget.cpp
      oculusrenderer.cpp
      oculuswindow.cpp
      steamvrrenderer.cpp
      steamvrwindow.cpp
      matrices.cpp)

  set(VIDEOSTITCH_BASE_HEADERS
      ${VIDEOSTITCH_BASE_HEADERS}
      matrices.hpp
      mirrorwidget.hpp
      oculusrenderer.hpp
      oculuswindow.hpp
      steamvrrenderer.hpp
      steamvrwindow.hpp
      matrices.hpp)
endif(WINDOWS)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
elseif(COMPILER_GCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

set(RESOURCES
    shaders.qrc)

add_library(${VS_BASE} SHARED ${VIDEOSTITCH_BASE_SOURCES} ${VIDEOSTITCH_BASE_HEADERS} ${RESOURCES})
add_dependencies(${VS_BASE} CheckVersionHeader)
target_compile_definitions(${VS_BASE} PRIVATE "VS_LIB_COMMON_COMPILATION")
set_property(TARGET ${VS_BASE} PROPERTY FOLDER "apps")
target_link_libraries(${VS_BASE} Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Widgets Qt5::Concurrent Qt5::Multimedia ${OPENGL_LIBRARIES} ${OVR} ${OPENVR} ${VS_DISCOVERY})
include_discovery_vs_headers(${VS_BASE})
include_lib_vs_headers(${VS_BASE})
add_cppcheck(${VS_BASE} VS)

