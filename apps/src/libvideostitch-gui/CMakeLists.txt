# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

# include($$_PRO_FILE_PWD_/../commonlib.pri)

# TODO maybe there'a a better way to handle this than a forced header

set(HEADERS
    base/ptvMerger.hpp
    caps/guistatecaps.hpp
    caps/signalcompressioncaps.hpp
    centralwidget/formats/codecs/basicmpegcodec.hpp
    centralwidget/formats/codecs/codec.hpp
    centralwidget/formats/codecs/codecfactory.hpp
    centralwidget/formats/codecs/h264codec.hpp
    centralwidget/formats/codecs/jpegcodec.hpp
    centralwidget/formats/codecs/mjpegcodec.hpp
    centralwidget/formats/codecs/mpeg2codec.hpp
    centralwidget/formats/codecs/mpeg4codec.hpp
    centralwidget/formats/codecs/mpeglikecodec.hpp
    centralwidget/formats/codecs/prorescodec.hpp
    centralwidget/formats/codecs/tiffcodec.hpp
    centralwidget/formats/codecs/trivialcodecs.hpp
    centralwidget/formats/extensionhandlers/extensionhandler.hpp
    centralwidget/formats/extensionhandlers/jpgextensionhandler.hpp
    centralwidget/formats/extensionhandlers/movextensionhandler.hpp
    centralwidget/formats/extensionhandlers/mp4extensionhandler.hpp
    centralwidget/formats/extensionhandlers/pamextensionhandler.hpp
    centralwidget/formats/extensionhandlers/pngextensionhandler.hpp
    centralwidget/formats/extensionhandlers/ppmextensionhandler.hpp
    centralwidget/formats/extensionhandlers/rawextensionhandler.hpp
    centralwidget/formats/extensionhandlers/stillimagehandler.hpp
    centralwidget/formats/extensionhandlers/tiffextensionhandler.hpp
    centralwidget/formats/extensionhandlers/yuv420pextensionhandler.hpp
    centralwidget/formats/format.hpp
    centralwidget/formats/formatfactory.hpp
    centralwidget/formats/movformat.hpp
    centralwidget/formats/mp4format.hpp
    centralwidget/formats/simpleformat.hpp
    centralwidget/gpulistwidget.hpp
    centralwidget/icentraltabwidget.hpp
    centralwidget/ifreezablewidget.hpp
    centralwidget/sourcewidget.hpp
    common.hpp
    dialogs/modalprogressdialog.hpp
    dialogs/resetdimensionsdialog.hpp
    mainwindow/downloader.hpp
    mainwindow/frameratecompute.hpp
    mainwindow/gpuinfoupdater.hpp
    mainwindow/LibLogHelpers.hpp
    mainwindow/msgboxhandlerhelper.hpp
    mainwindow/objectutil.hpp
    mainwindow/outputfilehandler.hpp
    mainwindow/packet.hpp
    mainwindow/processutility.hpp
    mainwindow/signalhandler.hpp
    mainwindow/statemanager.hpp
    mainwindow/stitchingwindow.hpp
    mainwindow/timeconverter.hpp
    mainwindow/uniqueqapplication.hpp
    mainwindow/versionHelper.hpp
    mainwindow/vscommandprocess.hpp
    mainwindow/vslocalserver.hpp
    mainwindow/vslocalsocket.hpp
    mainwindow/vssettings.hpp
    mainwindow/wintaskbarprogress.hpp
    mainwindow/ui_header/progressreporterwrapper.hpp
    utils/inputformat.hpp
    utils/outputformat.hpp
    utils/videocodecs.hpp
    utils/audiohelpers.hpp
    utils/bitratemodeenum.hpp
    utils/featurefilterer.hpp
    utils/gpuhelper.hpp
    utils/h264settingsenum.hpp
    utils/imagesorproceduralsonlyfilterer.hpp
    utils/inputlensenum.hpp
    utils/notonlyvideosfilterer.hpp
    utils/onevisualinputfilterer.hpp
    utils/panolensenum.hpp
    utils/pluginshelpers.hpp
    utils/smartenum.hpp
    utils/stereooutputenum.hpp
    utils/timersecondsenum.hpp
    utils/visibilityeventfilterer.hpp
    utils/widgetshelper.hpp
    utils/audioprocessors.hpp
    videostitcher/audioplayer.hpp
    videostitcher/backendInitializerProgressReporter.hpp
    videostitcher/frame.hpp
    videostitcher/mutableprojectdefinition.hpp
    videostitcher/presetsmanager.hpp
    videostitcher/projectdefinition.hpp
    videostitcher/stitchercontroller.hpp
    videostitcher/stitchercontrollerprogressreporter.hpp
    videostitcher/videostitcher.hpp
    widgets/aboutwindow.hpp
    widgets/autoselectspinbox.hpp
    widgets/crop/crop.hpp
    widgets/crop/cropcircleeditor.hpp
    widgets/crop/croprectangleeditor.hpp
    widgets/crop/cropshapeeditor.hpp
    widgets/crop/cropwidget.hpp
    widgets/crop/cropwindow.hpp
    widgets/crop/cropinputtab.hpp
    widgets/genericvideowidget.hpp
    widgets/levelmeter.hpp
    widgets/multivideowidget.hpp
    widgets/singlevideowidget.hpp
    widgets/deviceinteractivewidget.hpp
    widgets/devicevideowidget.hpp
    widgets/editabletimecode.hpp
    widgets/emorview.hpp
    widgets/logwidget.hpp
    widgets/panosizeselector.hpp
    widgets/rigwidget.hpp
    widgets/stylablewidget.hpp
    widgets/timewidget.hpp
    widgets/vignetteview.hpp
    widgets/vsgraphics.hpp
    widgets/vslistwidget.hpp
    widgets/vspathedit.hpp
    widgets/welcome/welcomescreenwidget.hpp
    widgets/welcome/projectselectionwidget.hpp
    widgets/welcome/softwarehelpwidget.hpp
    )

set(SOURCES
    base/ptvMerger.cpp
    caps/signalcompressioncaps.cpp
    centralwidget/ifreezablewidget.cpp
    centralwidget/formats/codecs/mpeglikecodec.cpp
    centralwidget/formats/formatfactory.cpp
    centralwidget/formats/movformat.cpp
    centralwidget/formats/mp4format.cpp
    centralwidget/formats/simpleformat.cpp
    centralwidget/formats/extensionhandlers/extensionhandler.cpp
    centralwidget/formats/codecs/basicmpegcodec.cpp
    centralwidget/formats/codecs/codecfactory.cpp
    centralwidget/formats/codecs/h264codec.cpp
    centralwidget/formats/codecs/mpeg4codec.cpp
    dialogs/modalprogressdialog.cpp
    dialogs/resetdimensionsdialog.cpp
    mainwindow/downloader.cpp
    mainwindow/msgboxhandlerhelper.cpp
    mainwindow/objectutil.cpp
    mainwindow/timeconverter.cpp
    mainwindow/packet.cpp
    mainwindow/processutility.cpp
    mainwindow/uniqueqapplication.cpp
    mainwindow/vscommandprocess.cpp
    mainwindow/vslocalserver.cpp
    mainwindow/vslocalsocket.cpp
    mainwindow/vssettings.cpp
    mainwindow/frameratecompute.cpp
    mainwindow/outputfilehandler.cpp
    mainwindow/statemanager.cpp
    mainwindow/stitchingwindow.cpp
    mainwindow/wintaskbarprogress.cpp
    mainwindow/versionHelper.cpp
    utils/bitratemodeenum.cpp
    utils/featurefilterer.cpp
    utils/gpuhelper.cpp
    utils/h264settingsenum.cpp
    utils/imagesorproceduralsonlyfilterer.cpp
    utils/inputlensenum.cpp
    utils/notonlyvideosfilterer.cpp
    utils/onevisualinputfilterer.cpp
    utils/panolensenum.cpp
    utils/videocodecs.cpp
    utils/visibilityeventfilterer.cpp
    utils/widgetshelper.cpp
    videostitcher/audioplayer.cpp
    videostitcher/backendInitializerProgressReporter.cpp
    videostitcher/globalcontroller.cpp
    videostitcher/mutableprojectdefinition.cpp
    videostitcher/presetsmanager.cpp
    videostitcher/projectdefinition.cpp
    videostitcher/stitchercontroller.cpp
    videostitcher/stitchercontrollerprogressreporter.cpp
    videostitcher/videostitcher.cpp
    widgets/aboutwindow.cpp
    widgets/vsgraphics.cpp
    widgets/editabletimecode.cpp
    widgets/levelmeter.cpp
    widgets/logwidget.cpp
    widgets/panosizeselector.cpp
    widgets/rigwidget.cpp
    widgets/timewidget.cpp
    widgets/devicevideowidget.cpp
    widgets/deviceinteractivewidget.cpp
    widgets/genericvideowidget.cpp
    widgets/multivideowidget.cpp
    widgets/emorview.cpp
    widgets/vignetteview.cpp
    centralwidget/gpulistwidget.cpp
    mainwindow/LibLogHelpers.cpp
    widgets/crop/cropshapeeditor.cpp
    widgets/crop/cropcircleeditor.cpp
    widgets/crop/croprectangleeditor.cpp
    widgets/crop/cropwidget.cpp
    widgets/crop/cropwindow.cpp
    widgets/crop/cropinputtab.cpp
    widgets/singlevideowidget.cpp
    widgets/welcome/welcomescreenwidget.cpp
    widgets/welcome/projectselectionwidget.cpp
    widgets/welcome/softwarehelpwidget.cpp
    utils/imagemanager.cpp
    utils/pluginshelpers.cpp
    centralwidget/formats/codecs/mjpegcodec.cpp
    centralwidget/sourcewidget.cpp
    utils/stereooutputenum.cpp
    utils/sourcewidgetlayoututil.cpp
    )

set(FORMS
    centralwidget/sourcewidget.ui
    dialogs/modalprogressdialog.ui
    dialogs/resetdimensionsdialog.ui
    widgets/aboutwindow.ui
    widgets/logwidget.ui
    widgets/panosizeselector.ui
    widgets/rigwidget.ui
    widgets/timewidget.ui
    widgets/crop/cropwidget.ui
    widgets/crop/cropinputtab.ui
    widgets/welcome/welcomescreenwidget.ui
    widgets/welcome/projectselectionwidget.ui
    widgets/welcome/softwarehelpwidget.ui)

set(RESOURCES
    libvideostitch-gui.qrc
    ../libvideostitch-base/shaders.qrc
    ../resources/common-resources.qrc
    # add the qss resources to be able to search in it
    assets/qss/style_variables.ini
    ../resources/style/vs_common.qss
    ../resources/style/common_style_variables.ini)

qt5_wrap_ui(UI_GENERATED_HEADERS ${FORMS})

if(WINDOWS)
  set(SOURCES
      ${SOURCES}
      mainwindow/stitcheroculuswindow.cpp
      mainwindow/stitchersteamvrwindow.cpp
      )
  set(HEADERS
      ${HEADERS}
      mainwindow/stitcheroculuswindow.hpp
      mainwindow/stitchersteamvrwindow.hpp
      )
endif(WINDOWS)

list(GET UI_GENERATED_HEADERS 0 FIRST_GEN_HEADER)
get_filename_component(LIB_VS_GUI_GEN_HEADER_DIR ${FIRST_GEN_HEADER} DIRECTORY)
set(LIB_VS_GUI_GEN_HEADER_DIR ${LIB_VS_GUI_GEN_HEADER_DIR} PARENT_SCOPE)

add_library(${VS_GUI} SHARED ${SOURCES} ${HEADERS} ${UI_GENERATED_HEADERS} ${RESOURCES})
target_compile_definitions(${VS_GUI} PRIVATE "VS_LIB_GUI_COMPILATION")
set_property(TARGET ${VS_GUI} PROPERTY FOLDER "apps")
set_target_properties(${VS_GUI} PROPERTIES COMPILE_FLAGS "${COMMON_FLAGS}")

target_include_directories(${VS_GUI} PRIVATE mainwindow/ui_header)

include_lib_vs_headers(${VS_GUI})
include_discovery_vs_headers(${VS_GUI})

set_property(TARGET ${VS_GUI} PROPERTY CXX_STANDARD 14)
target_link_libraries(
    ${VS_GUI}
    PRIVATE
    ${VS_BASE}
    ${VS_DISCOVERY}
    Qt5::Concurrent
    Qt5::Core
    Qt5::Gui
    Qt5::Multimedia
    Qt5::Network
    Qt5::OpenGL
    Qt5::Widgets
    )
link_target_to_libvideostitch(${VS_GUI})
add_cppcheck(${VS_GUI} VS)

