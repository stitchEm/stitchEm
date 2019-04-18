# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

set(STUDIO_SOURCES
    src/centralwidget/centralstackedwidget.cpp
    src/centralwidget/outputtabwidget.cpp
    src/centralwidget/processtabwidget.cpp
    src/centralwidget/processtab/iprocesswidget.cpp
    src/centralwidget/processtab/outputfileprocess.cpp
    src/centralwidget/processtab/videoprocess.cpp
    src/centralwidget/processtab/audioprocess.cpp
    src/dialogs/extractdialog.cpp
    src/dialogs/jumpdialog.cpp
    src/dialogs/preferencesdialog.cpp
    src/dialogs/processprogressdialog.cpp
    src/dialogs/shortcutdialog.cpp
    src/main/main.cpp
    src/mainwindow/mainwindow.cpp
    src/videostitcher/postprodmutableprojectdefinition.cpp
    src/videostitcher/postprodprojectdefinition.cpp
    src/widgets/advancedblendingwidget.cpp
    src/widgets/blendingmaskwidget.cpp
    src/widgets/calibrationwidget.cpp
    src/widgets/computationwidget.cpp
    src/widgets/curvestreewidget.cpp
    src/widgets/exposurewidget.cpp
    src/widgets/interactivetabwidget.cpp
    src/widgets/seekbar.cpp
    src/widgets/stabilizationwidget.cpp
    src/widgets/synchronizationwidget.cpp
    src/widgets/outputconfigurationwidget.cpp
    src/widgets/timeline/timelinecontainer.cpp
    src/widgets/timeline/timelineticks.cpp
    src/widgets/timeline/timeline.cpp
    src/widgets/timesequence.cpp
    src/widgets/timeline/constrainedgraphicsitemgroup.cpp
    src/widgets/timeline/curvegraphicsitem.cpp
    src/widgets/timeline/timelinescene.cpp
    src/commands/synchronizationoffsetschangedcommand.cpp
    src/commands/calibrationappliedcommand.cpp
    src/commands/blendingmaskappliedcommand.cpp
    src/commands/externcalibrationappliedcommand.cpp
    src/commands/outputsizechangedcommand.cpp
    src/commands/panochangedcommand.cpp
    src/commands/blenderchangedcommand.cpp
    src/commands/advancedblenderchangedcommand.cpp
    src/commands/spherescalechangedcommand.cpp
    src/commands/stabilizationcomputedcommand.cpp
    src/videostitcher/postprodstitchercontroller.cpp
    src/videostitcher/globalpostprodcontroller.cpp
    src/commands/workingareachangedcommand.cpp
    src/commands/exposureappliedcommand.cpp
    src/commands/photometriccalibrationappliedcommand.cpp
    src/commands/orientationchangedcommand.cpp
    src/commands/projectionchangedcommand.cpp
    src/mainwindow/shortcutmanager.cpp
    src/widgets/photometriccalibrationwidget.cpp
    src/widgets/stereoconfigurationwidget.cpp
    src/mainwindow/postprodsettings.cpp)

set(STUDIO_HEADERS
    src/centralwidget/centralstackedwidget.hpp
    src/centralwidget/outputtabwidget.hpp
    src/centralwidget/processtabwidget.hpp
    src/centralwidget/processtab/iprocesswidget.hpp
    src/centralwidget/processtab/outputfileprocess.hpp
    src/centralwidget/processtab/videoprocess.hpp
    src/centralwidget/processtab/audioprocess.hpp
    src/commands/advancedblenderchangedcommand.hpp
    src/commands/blenderchangedcommand.hpp
    src/commands/spherescalechangedcommand.hpp
    src/commands/calibrationappliedcommand.hpp
    src/commands/exposureappliedcommand.hpp
    src/commands/photometriccalibrationappliedcommand.hpp
    src/commands/externcalibrationappliedcommand.hpp
    src/commands/orientationchangedcommand.hpp
    src/commands/outputsizechangedcommand.hpp
    src/commands/panochangedcommand.hpp
    src/commands/projectionchangedcommand.hpp
    src/commands/stabilizationcomputedcommand.hpp
    src/commands/synchronizationoffsetschangedcommand.hpp
    src/commands/undoHelpers.hpp
    src/commands/workingareachangedcommand.hpp
    src/dialogs/extractdialog.hpp
    src/dialogs/jumpdialog.hpp
    src/dialogs/preferencesdialog.hpp
    src/dialogs/processprogressdialog.hpp
    src/dialogs/shortcutdialog.hpp
    src/mainwindow/crasher.hpp
    src/mainwindow/mainwindow.hpp
    src/mainwindow/postprodsettings.hpp
    src/mainwindow/shortcutmanager.hpp
    src/videostitcher/postprodmutableprojectdefinition.hpp
    src/videostitcher/postprodprojectdefinition.hpp
    src/videostitcher/postprodstitchercontroller.hpp
    src/widgets/itoolwidget.hpp
    src/widgets/advancedblendingwidget.hpp
    src/widgets/blendingmaskwidget.hpp
    src/widgets/calibrationwidget.hpp
    src/widgets/computationwidget.hpp
    src/widgets/curvestreewidget.hpp
    src/widgets/exposurewidget.hpp
    src/widgets/interactivetabwidget.hpp
    src/widgets/photometriccalibrationwidget.hpp
    src/widgets/seekbar.hpp
    src/widgets/stabilizationwidget.hpp
    src/widgets/stereoconfigurationwidget.hpp
    src/widgets/synchronizationwidget.hpp
    src/widgets/outputconfigurationwidget.hpp
    src/widgets/timeline/constrainedgraphicsitemgroup.hpp
    src/widgets/timeline/curvegraphicsitem.hpp
    src/widgets/timeline/timeline.hpp
    src/widgets/timesequence.hpp
    src/widgets/timeline/timelinecontainer.hpp
    src/widgets/timeline/timelinescene.hpp
    src/widgets/timeline/timelineticks.hpp
    )

set(STUDIO_FORMS
    src/centralwidget/outputtabwidget.ui
    src/centralwidget/processtabwidget.ui
    src/centralwidget/processtab/outputfileprocess.ui
    src/centralwidget/processtab/videoprocess.ui
    src/centralwidget/processtab/audioprocess.ui
    src/dialogs/extractdialog.ui
    src/dialogs/preferencesdialog.ui
    src/mainwindow/mainwindow.ui
    src/widgets/seekbar.ui
    src/dialogs/jumpdialog.ui
    src/dialogs/welcomedialog.ui
    src/dialogs/shortcutdialog.ui
    src/widgets/synchronizationwidget.ui
    src/widgets/exposurewidget.ui
    src/widgets/interactivetabwidget.ui
    src/widgets/timeline/timelinecontainer.ui
    src/dialogs/processprogressdialog.ui
    src/widgets/advancedblendingwidget.ui
    src/widgets/blendingmaskwidget.ui
    src/widgets/calibrationwidget.ui
    src/widgets/stabilizationwidget.ui
    src/widgets/outputconfigurationwidget.ui
    src/widgets/timesequence.ui
    src/widgets/photometriccalibrationwidget.ui
    src/widgets/stereoconfigurationwidget.ui
    )


set(RESOURCES
    videostitch-studio-gui.qrc
    # add the qss resources to be able to search in it
    videostitch-studio.qss)

qt5_wrap_ui(UI_GENERATED_HEADERS ${STUDIO_FORMS})

include_directories(src)

if(BUILD_MACOSX_BUNDLE AND APPLE)
  # NOTE: Don't include the path in MACOSX_BUNDLE_ICON_FILE -- this is
  # the property added to Info.plist
  set(MACOSX_BUNDLE_ICON_FILE videostitch.icns)

  # And this part tells CMake where to find and install the file itself
  set(Studio_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/vs/videostitch.icns)
  set_source_files_properties(${Studio_ICON} PROPERTIES
                              MACOSX_PACKAGE_LOCATION "Resources")

  set(VS_STUDIO_BUNDLE "VideoStitch-Studio")
  set(BIN_NAME "${VS_STUDIO_BUNDLE}")

  add_executable(${VS_STUDIO_BUNDLE} MACOSX_BUNDLE ${STUDIO_SOURCES} ${STUDIO_HEADERS} ${UI_GENERATED_HEADERS} ${RESOURCES})
  set_target_properties(${VS_STUDIO_BUNDLE} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${PROJECT_SOURCE_DIR}/dmg/videostitch-studio.plist )

  add_dependencies(${VS_STUDIO_BUNDLE} batchstitcher)
  add_vs_IO_deps(${VS_STUDIO_BUNDLE})
  include_lib_vs_headers(${VS_STUDIO_BUNDLE})
  include_discovery_vs_headers(${VS_STUDIO_BUNDLE})
  target_include_directories(${VS_STUDIO_BUNDLE} PRIVATE ${LIB_VS_GUI_GEN_HEADER_DIR})
  set_property(TARGET ${VS_STUDIO_BUNDLE} PROPERTY CXX_STANDARD 14)
  target_link_libraries(${VS_STUDIO_BUNDLE} PRIVATE Qt5::Concurrent Qt5::Gui Qt5::Multimedia Qt5::OpenGL Qt5::Widgets ${VS_GUI} ${VS_BASE} ${CORE_FOUNDATION} ${VS_DISCOVERY})
  link_target_to_libvideostitch(${VS_STUDIO_BUNDLE})
else()

  # ----------------------------------------------------------------------------
  # Studio output directories
  # ----------------------------------------------------------------------------
  # Set the resources file for icon
  set(STUDIO_RC studio.rc)

  # Set binary output dir for the generic single-config case (e.g. make, ninja)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_STUDIO_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_STUDIO_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_STUDIO_DIR})

  # Set binary output dir for multi-config builds (e.g. MSVC, Xcode)
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_STUDIO_DIR_${OUTPUTCONFIG_UP}})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_STUDIO_DIR_${OUTPUTCONFIG_UP}})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_STUDIO_DIR_${OUTPUTCONFIG_UP}})
  endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

  # ----------------------------------------------------------------------------

  set(VS_STUDIO_BIN "videostitch-studio")
  set(BIN_NAME "${VS_STUDIO_BIN}")
  add_executable(${VS_STUDIO_BIN} WIN32 ${STUDIO_SOURCES} ${STUDIO_HEADERS} ${UI_GENERATED_HEADERS} ${RESOURCES} ${STUDIO_RC})

  add_dependencies(${VS_STUDIO_BIN} batchstitcher)
  add_vs_IO_deps(${VS_STUDIO_BIN})
  include_lib_vs_headers(${VS_STUDIO_BIN})
  include_discovery_vs_headers(${VS_STUDIO_BIN})
  target_include_directories(${VS_STUDIO_BIN} PRIVATE ${LIB_VS_GUI_GEN_HEADER_DIR})
  set_property(TARGET ${VS_STUDIO_BIN} PROPERTY CXX_STANDARD 14)
  target_link_libraries(${VS_STUDIO_BIN} PRIVATE Qt5::Concurrent Qt5::Gui Qt5::Multimedia Qt5::OpenGL Qt5::Widgets ${VS_GUI} ${VS_BASE} ${CORE_FOUNDATION} ${VS_DISCOVERY})
  link_target_to_libvideostitch(${VS_STUDIO_BIN})
endif()

target_compile_definitions(${BIN_NAME} PRIVATE "APPS_VIDEOSTITCH_VERSION=\"${STUDIO_VERSION}\"")
set_target_properties(${BIN_NAME} PROPERTIES COMPILE_FLAGS "${COMMON_FLAGS}")

# Add filter to MSVS project
assign_source_group(${STUDIO_SOURCES} ${STUDIO_HEADERS} ${STUDIO_FORMS})
add_cppcheck(${BIN_NAME} VS)

