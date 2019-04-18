
# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

set(SOURCES
    main.cpp
    tasktable.cpp
    batchertask.cpp
    batchwindow.cpp
    logdialog.cpp)

set(HEADERS
    tasktable.hpp
    batchertask.hpp
    autoelidelabel.hpp
    batchwindow.hpp
    logdialog.hpp)

set(FORMS
    batchwindow.ui
    logdialog.ui)

set(RESOURCES
    batchstitcher.qrc
    # add the qss resources to be able to search in it
    resources/batchstitcher.qss)

qt5_wrap_ui(UI_GENERATED_HEADERS ${FORMS})
set(BIN_NAME "batchstitcher")

# Set the resources file for icon
set(BATCH_RC batch.rc)

add_executable(${BIN_NAME} WIN32 MACOSX_BUNDLE ${SOURCES} ${HEADERS} ${UI_GENERATED_HEADERS} ${RESOURCES} ${BATCH_RC})
set_property(TARGET ${BIN_NAME} PROPERTY FOLDER "apps")
include_lib_vs_headers(${BIN_NAME})
include_discovery_vs_headers(${BIN_NAME})
add_dependencies(${BIN_NAME} videostitch-cmd)
set_property(TARGET ${BIN_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${BIN_NAME} PRIVATE Qt5::Gui Qt5::Network Qt5::Widgets ${VS_GUI} ${VS_BASE} ${VS_DISCOVERY})

link_target_to_libvideostitch(${BIN_NAME})

set_target_properties(${BIN_NAME} PROPERTIES COMPILE_FLAGS "${COMMON_FLAGS}")
