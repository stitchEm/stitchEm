# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

set(VAHANA_SOURCES
    main.cpp
    src/animations/animatedwidget.cpp
    src/animations/fadeanimation.cpp
    src/animations/ianimation.cpp
    src/authenticationwidget.cpp
    src/calibration/calibrationactioncontroller.cpp
    src/calibration/calibrationsnapcounter.cpp
    src/calibration/calibrationupdatecontroller.cpp
    src/calibration/calibrationworkflowpage.cpp
    src/calibration/cropworkflowpage.cpp
    src/calibration/rigworkflowpage.cpp
    src/configurations/ajainputconfiguration.cpp
    src/configurations/ajaoutputconfiguration.cpp
    src/configurations/audioinputconfigurationwidget.cpp
    src/configurations/basertmpserviceconfiguration.cpp
    src/configurations/capturecardinputconfiguration.cpp
    src/configurations/configexposurewidget.cpp
    src/configurations/configoutputaudio.cpp
    src/configurations/configoutputswidget.cpp
    src/configurations/configpanorama.cpp
    src/configurations/configurationinputprocedural.cpp
    src/configurations/configurationinputmedia.cpp
    src/configurations/configurationoutputhdd.cpp
    src/configurations/configurationoutputsdi.cpp
    src/configurations/configurationoutputstreaming.cpp
    src/configurations/configurationtablewidget.cpp
    src/configurations/iconfigurationvalue.cpp
    src/configurations/inputconfigurationwidget.cpp
    src/configurations/istreamingserviceconfiguration.cpp
    src/configurations/outputconfigurationwidget.cpp
    src/configurations/resizepanoramawidget.cpp
    src/configurations/rigconfigurationwidget.cpp
    src/configurationtabwidget.cpp
    src/credentialfactory.cpp
    src/credentialmodel.cpp
    src/cropinputcontroller.cpp
    src/dialogbackground.cpp
    src/exposure/exposureactioncontroller.cpp
    src/exposure/liveexposure.cpp
    src/generic/backgroundcontainer.cpp
    src/generic/genericdialog.cpp
    src/generic/genericloader.cpp
    src/generic/generictablewidget.cpp
    src/generic/workflowdialog.cpp
    src/generic/workflowpage.cpp
    src/guiconstants.cpp
    src/inputeyeselector.cpp
    src/inputtypewidget.cpp
    src/interactivetabwidget.cpp
    src/livesettings.cpp
    src/logdialog.cpp
    src/maintabwidget.cpp
    src/mainwindow.cpp
    src/multiinputconfiguration.cpp
    src/newinputitemwidget.cpp
    src/newprojectnamedialog.cpp
    src/outputcontrolspanel.cpp
    src/outputdetailwidget.cpp
    src/outputscontroller.cpp
    src/outputtabwidget.cpp
    src/plugin/inputplugincontroller.cpp
    src/plugin/newoutputwidget.cpp
    src/plugin/outputplugincontroller.cpp
    src/plugin/pluginscontroller.cpp
    src/projectworkwidget.cpp
    src/resizepanoramadialog.cpp
    src/sectionmodel.cpp
    src/sectionview.cpp
    src/settings/configcalibrationwidget.cpp
    src/settings/generalsettingswidget.cpp
    src/settings/oculussettingswidget.cpp
    src/settings/settinggpu.cpp
    src/sourcestabwidget.cpp
    src/topinformationbarwidget.cpp
    src/utils/displaymode.cpp
    src/utils/maintabs.cpp
    src/utils/pixelformat.cpp
    src/videostitcher/audioconfiguration.cpp
    src/videostitcher/capturecardliveinput.cpp
    src/videostitcher/encodedoutputhelpers.cpp
    src/videostitcher/globallivecontroller.cpp
    src/videostitcher/liveaudio.cpp
    src/videostitcher/liveinputaja.cpp
    src/videostitcher/liveinputdecklink.cpp
    src/videostitcher/liveinputfactory.cpp
    src/videostitcher/liveinputfile.cpp
    src/videostitcher/liveinputmagewell.cpp
    src/videostitcher/liveinputximea.cpp
    src/videostitcher/liveinputprocedural.cpp
    src/videostitcher/liveinputstream.cpp
    src/videostitcher/liveinputv4l2.cpp
    src/videostitcher/livemutableprojectdefinition.cpp
    src/videostitcher/liveoutputaja.cpp
    src/videostitcher/liveoutputcustom.cpp
    src/videostitcher/liveoutputdecklink.cpp
    src/videostitcher/liveoutputfactory.cpp
    src/videostitcher/liveoutputfile.cpp
    src/videostitcher/liveoutputlist.cpp
    src/videostitcher/liveoutputoculus.cpp
    src/videostitcher/liveoutputrtmp.cpp
    src/videostitcher/liveoutputsdi.cpp
    src/videostitcher/liveprojectdefinition.cpp
    src/videostitcher/livestitchercontroller.cpp
    src/videostitcher/liveaudioprocessfactory.cpp
    src/videostitcher/liveaudioprocessordelay.cpp
    src/widgets/audioprocessors/audioprocessorswidget.cpp
    src/widgets/audioprocessors/newaudioprocessorwidget.cpp
    src/widgets/audioprocessors/audioprocessorconfigurationwidget.cpp
    src/widgets/audioprocessors/delayprocessorwidget.cpp
    src/widgets/audioprocessors/audioprocessdetailwidget.cpp
    src/widgetsmanager.cpp
    )

set(VAHANA_HEADERS
    src/animations/animatedwidget.hpp
    src/animations/fadeanimation.hpp
    src/animations/ianimation.hpp
    src/authenticationwidget.hpp
    src/calibration/calibrationactioncontroller.hpp
    src/calibration/calibrationsnapcounter.hpp
    src/calibration/calibrationupdatecontroller.hpp
    src/calibration/calibrationworkflowpage.hpp
    src/calibration/cropworkflowpage.hpp
    src/calibration/rigworkflowpage.hpp
    src/configurations/ajainputconfiguration.hpp
    src/configurations/ajaoutputconfiguration.hpp
    src/configurations/audioinputconfigurationwidget.hpp
    src/configurations/basertmpserviceconfiguration.hpp
    src/configurations/capturecardinputconfiguration.hpp
    src/configurations/configexposurewidget.hpp
    src/configurations/configoutputaudio.hpp
    src/configurations/configoutputswidget.hpp
    src/configurations/configpanorama.hpp
    src/configurations/configurationinputprocedural.hpp
    src/configurations/configurationinputmedia.hpp
    src/configurations/configurationoutputhdd.hpp
    src/configurations/configurationoutputsdi.hpp
    src/configurations/configurationoutputstreaming.hpp
    src/configurations/configurationtablewidget.hpp
    src/configurations/iconfigurationvalue.hpp
    src/configurations/inputconfigurationwidget.hpp
    src/configurations/istreamingserviceconfiguration.hpp
    src/configurations/outputconfigurationwidget.hpp
    src/configurations/resizepanoramawidget.hpp
    src/configurations/rigconfigurationwidget.hpp
    src/configurationtabwidget.hpp
    src/credentialfactory.hpp
    src/credentialmodel.hpp
    src/cropinputcontroller.hpp
    src/dialogbackground.hpp
    src/exposure/exposureactioncontroller.hpp
    src/exposure/liveexposure.hpp
    src/generic/backgroundcontainer.hpp
    src/generic/genericdialog.hpp
    src/generic/genericloader.hpp
    src/generic/generictablewidget.hpp
    src/generic/workflowdialog.hpp
    src/generic/workflowpage.hpp
    src/guiconstants.hpp
    src/inputeyeselector.hpp
    src/inputtypewidget.hpp
    src/interactivetabwidget.hpp
    src/livesettings.hpp
    src/logdialog.hpp
    src/maintabwidget.hpp
    src/mainwindow.hpp
    src/multiinputconfiguration.hpp
    src/newinputitemwidget.hpp
    src/newprojectnamedialog.hpp
    src/outputcontrolspanel.hpp
    src/outputdetailwidget.hpp
    src/outputscontroller.hpp
    src/outputtabwidget.hpp
    src/plugin/inputplugincontroller.hpp
    src/plugin/newoutputwidget.hpp
    src/plugin/outputplugincontroller.hpp
    src/plugin/pluginscontroller.hpp
    src/plugin/staticoutputs.hpp
    src/projectworkwidget.hpp
    src/resizepanoramadialog.hpp
    src/sectionmodel.hpp
    src/sectionview.hpp
    src/settings/configcalibrationwidget.hpp
    src/settings/generalsettingswidget.hpp
    src/settings/iappsettings.hpp
    src/settings/oculussettingswidget.hpp
    src/settings/settinggpu.hpp
    src/sourcestabwidget.hpp
    src/topinformationbarwidget.hpp
    src/utils/displaymode.hpp
    src/utils/pixelformat.hpp
    src/videostitcher/audioconfiguration.hpp
    src/videostitcher/capturecardliveinput.hpp
    src/videostitcher/encodedoutputhelpers.hpp
    src/videostitcher/globallivecontroller.cpp
    src/videostitcher/liveaudio.hpp
    src/videostitcher/liveinputaja.hpp
    src/videostitcher/liveinputdecklink.hpp
    src/videostitcher/liveinputfactory.hpp
    src/videostitcher/liveinputfile.hpp
    src/videostitcher/liveinputmagewell.hpp
    src/videostitcher/liveinputximea.hpp
    src/videostitcher/liveinputprocedural.hpp
    src/videostitcher/liveinputstream.hpp
    src/videostitcher/liveinputv4l2.hpp
    src/videostitcher/livemutableprojectdefinition.hpp
    src/videostitcher/liveoutputaja.hpp
    src/videostitcher/liveoutputcustom.hpp
    src/videostitcher/liveoutputdecklink.hpp
    src/videostitcher/liveoutputfactory.hpp
    src/videostitcher/liveoutputfile.hpp
    src/videostitcher/liveoutputlist.hpp
    src/videostitcher/liveoutputoculus.hpp
    src/videostitcher/liveoutputrtmp.hpp
    src/videostitcher/liveoutputsdi.hpp
    src/videostitcher/liveprojectdefinition.hpp
    src/videostitcher/livestitchercontroller.hpp
    src/videostitcher/liveaudioprocessfactory.hpp
    src/videostitcher/liveaudioprocessordelay.hpp
    src/widgets/audioprocessors/audioprocessorswidget.hpp
    src/widgets/audioprocessors/newaudioprocessorwidget.hpp
    src/widgets/audioprocessors/audioprocessorconfigurationwidget.hpp
    src/widgets/audioprocessors/delayprocessorwidget.hpp
    src/widgets/audioprocessors/audioprocessdetailwidget.hpp
    src/widgetsmanager.hpp)

set(VAHANA_FORMS
    src/authenticationwidget.ui
    src/calibration/calibrationworkflowpage.ui
    src/calibration/cropworkflowpage.ui
    src/calibration/rigworkflowpage.ui
    src/configurations/ajainputconfiguration.ui
    src/configurations/ajaoutputconfiguration.ui
    src/configurations/audioinputconfigurationwidget.ui
    src/configurations/basertmpserviceconfiguration.ui
    src/configurations/capturecardinputconfiguration.ui
    src/configurations/configexposurewidget.ui
    src/configurations/configoutputaudio.ui
    src/configurations/configurationinputprocedural.ui
    src/configurations/configurationinputmedia.ui
    src/configurations/configurationoutputstreaming.ui
    src/configurations/forms/configinputswidget.ui
    src/configurations/forms/configoutputswidget.ui
    src/configurations/forms/configurationoutputhdd.ui
    src/configurations/forms/configurationoutputsdi.ui
    src/configurations/forms/configurationpanorama.ui
    src/configurations/forms/configurationtablewidget.ui
    src/configurations/resizepanoramawidget.ui
    src/configurations/rigconfigurationwidget.ui
    src/forms/configurationtabwidget.ui
    src/forms/interactivetabwidget.ui
    src/forms/mainwindow.ui
    src/forms/newinputitemwidget.ui
    src/forms/outputcontrolspanel.ui
    src/forms/outputdetailwidget.ui
    src/forms/outputtabwidget.ui
    src/forms/projectworkwidget.ui
    src/forms/sourcestabwidget.ui
    src/forms/topinformationbarwidget.ui
    src/generic/backgroundcontainer.ui
    src/generic/forms/genericdialog.ui
    src/generic/workflowdialog.ui
    src/widgets/audioprocessors/audioprocessorswidget.ui
    src/widgets/audioprocessors/newaudioprocessorwidget.ui
    src/widgets/audioprocessors/delayprocessorwidget.ui
    src/widgets/audioprocessors/audioprocessdetailwidget.ui
    src/inputeyeselector.ui
    src/inputtypewidget.ui
    src/multiinputconfiguration.ui
    src/plugin/newoutputwidget.ui
    src/settings/configcalibrationwidget.ui
    src/settings/generalsettingswidget.ui
    src/settings/oculussettingswidget.ui
    src/settings/settinggpu.ui
    )

if(ENABLE_YOUTUBE_OUTPUT)
  set(VAHANA_SOURCES
      ${VAHANA_SOURCES}
      src/configurations/youtubebaseconfiguration.cpp
      src/configurations/youtubeconfiguration.cpp
      src/configurations/youtubetableview.cpp
      src/videostitcher/liveoutputyoutube.cpp
      src/youtubebroadcastmodel.cpp
      src/googlecredentialmodel.cpp
      src/googleauthenticationmanager.cpp)
  set(VAHANA_HEADERS
      ${VAHANA_HEADERS}
      src/configurations/youtubebaseconfiguration.hpp
      src/configurations/youtubeconfiguration.hpp
      src/configurations/youtubetableview.hpp
      src/videostitcher/liveoutputyoutube.hpp
      src/youtubebroadcastmodel.hpp
      src/googlecredentialmodel.hpp
      src/googleauthenticationmanager.hpp)
  set(VAHANA_FORMS
      ${VAHANA_FORMS}
      src/configurations/youtubebaseconfiguration.ui
      src/configurations/youtubeconfiguration.ui)
endif()

if(WINDOWS)
  set(VAHANA_SOURCES
      ${VAHANA_SOURCES}
      # steam VR
      src/videostitcher/liveoutputsteamvr.cpp
      )

  set(VAHANA_HEADERS
      ${VAHANA_HEADERS}
      # steam VR
      src/videostitcher/liveoutputsteamvr.hpp
      )
endif(WINDOWS)

set(RESOURCES
    videostitch-live-gui.qrc
    # add the qss resources to be able to search in it
    resources/style_variables.ini
    resources/vahana-vr.qss)

qt5_wrap_ui(UI_GENERATED_HEADERS ${VAHANA_FORMS})

# ----------------------------------------------------------------------------
# Vahana VR output directories
# ----------------------------------------------------------------------------
# Set binary output dir for the generic single-config case (e.g. make, ninja)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_VAHANA_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_VAHANA_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_VAHANA_DIR})

# Set binary output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_DIR_${OUTPUTCONFIG_UP}})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_DIR_${OUTPUTCONFIG_UP}})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_VAHANA_DIR_${OUTPUTCONFIG_UP}})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# ----------------------------------------------------------------------------

include_directories(src)
include_directories(src/animations)
include_directories(src/calibration)
include_directories(src/configurations)
include_directories(src/exposure)
include_directories(src/generic)
include_directories(src/plugin)
include_directories(src/utils)
include_directories(src/videostitcher)

if(ENABLE_YOUTUBE_OUTPUT)
  if(WINDOWS)
    find_debug_and_optimized_library(GGYOUTUBE  "youtubeapi/debug" "google_youtube_api"       "youtubeapi/release" "google_youtube_api")
    find_debug_and_optimized_library(GGJSON_CPP "googleapis/debug" "googleapis_jsoncpp"       "googleapis/release" "googleapis_jsoncpp")
    find_debug_and_optimized_library(JSON       "json/debug"       "lib_json"                 "json/release"       "lib_json")
    find_debug_and_optimized_library(GGUTILS    "googleapis/debug" "googleapis_utils"         "googleapis/release" "googleapis_utils")
    find_debug_and_optimized_library(GGINTERNAL "googleapis/debug" "googleapis_internal"      "googleapis/release" "googleapis_internal")
    find_debug_and_optimized_library(GGHTTP     "googleapis/debug" "googleapis_http"          "googleapis/release" "googleapis_http")
    find_debug_and_optimized_library(GGOAUTH    "googleapis/debug" "googleapis_oauth2"        "googleapis/release" "googleapis_oauth2")
    find_debug_and_optimized_library(GGCURLHTTP "googleapis/debug" "googleapis_curl_http"     "googleapis/release" "googleapis_curl_http")
    find_debug_and_optimized_library(GGSSL      "googleapis/debug" "googleapis_openssl_codec" "googleapis/release" "googleapis_openssl_codec")
    find_debug_and_optimized_library(GGJSON     "googleapis/debug" "googleapis_json"          "googleapis/release" "googleapis_json")
    find_debug_and_optimized_library(GGJSONPB   "googleapis/debug" "googleapis_jsonplayback"  "googleapis/release" "googleapis_jsonplayback")
    find_debug_and_optimized_library(GGSCRIBES  "googleapis/debug" "googleapis_scribes"       "googleapis/release" "googleapis_scribes")
    find_library(CURL NAMES "libcurl" PATHS "${CMAKE_EXTERNAL_LIB}/curl")
  endif(WINDOWS)

  if(LINUX)
    find_library(JSON_CPP NAMES "jsoncpp" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGYOUTUBE NAMES "google_youtube_api" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGJSON_CPP NAMES "googleapis_jsoncpp" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGUTILS NAMES "googleapis_utils" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGINTERNAL NAMES "googleapis_internal" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGHTTP NAMES "googleapis_http" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGOAUTH NAMES "googleapis_oauth2" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGCURLHTTP NAMES "googleapis_curl_http" PATHS ${CMAKE_EXTERNAL_LIB})
    find_library(GGSSL NAMES "googleapis_openssl_codec" PATHS ${CMAKE_EXTERNAL_LIB})
  endif(LINUX)
endif(ENABLE_YOUTUBE_OUTPUT)

set(VAHANA_RC vahana.rc) # Set the resources file for icon
set(VS_VAHANA_BIN "VahanaVR")
add_executable(${VS_VAHANA_BIN} WIN32 ${VAHANA_SOURCES} ${VAHANA_HEADERS} ${UI_GENERATED_HEADERS} ${RESOURCES} ${VAHANA_RC})
add_vs_IO_deps(${VS_VAHANA_BIN})

if(WINDOWS)
  target_compile_definitions(${VS_VAHANA_BIN} PRIVATE "COMPILER_MSVC")
  target_include_directories(${VS_VAHANA_BIN} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/openvr")
endif(WINDOWS)
target_compile_definitions(${VS_VAHANA_BIN} PRIVATE "QT_PLUGIN")
target_compile_definitions(${VS_VAHANA_BIN} PRIVATE "APPS_VIDEOSTITCH_VERSION=\"${VAHANA_VERSION}\"")
if(ENABLE_YOUTUBE_OUTPUT)
  target_compile_definitions(${VS_VAHANA_BIN} PRIVATE "ENABLE_YOUTUBE_OUTPUT")

  # flags for google and youtube deps
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /J /wd4267")
  elseif(COMPILER_GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs -Wno-ignored-qualifiers")
  endif()
endif()

if(COMPILER_GCC)
  # FIXME: strict-aliasing (src/calibration/calibrationupdatecontroller.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
endif()

link_target_to_libvideostitch(${VS_VAHANA_BIN})
include_lib_vs_headers(${VS_VAHANA_BIN})
include_discovery_vs_headers(${VS_VAHANA_BIN})
target_include_directories(${VS_VAHANA_BIN} PRIVATE ${LIB_VS_GUI_GEN_HEADER_DIR})
target_include_directories(${VS_VAHANA_BIN} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include")
target_include_directories(${VS_VAHANA_BIN} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/googleapis")
target_include_directories(${VS_VAHANA_BIN} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/jsoncpp")
target_link_libraries(
  ${VS_VAHANA_BIN}
  PRIVATE
  Qt5::Concurrent
  Qt5::Gui
  Qt5::Multimedia
  Qt5::OpenGL
  Qt5::Widgets
  ${VS_GUI}
  ${VS_BASE}
  ${VS_DISCOVERY}
  )
if(ENABLE_YOUTUBE_OUTPUT)
  target_link_libraries(
    ${VS_VAHANA_BIN}
    PRIVATE
    ${JSON_CPP}
    ${GGYOUTUBE}
    ${GGJSON_CPP}
    ${GGUTILS}
    ${GGINTERNAL}
    ${GGHTTP}
    ${GGOAUTH}
    ${GGCURLHTTP}
    ${GGSSL}
    )
endif()
if(WINDOWS)
  if(ENABLE_YOUTUBE_OUTPUT)
    target_include_directories(${VS_VAHANA_BIN} PRIVATE "${CMAKE_EXTERNAL_DEPS}/include/youtubeapi")
    target_link_libraries(${VS_VAHANA_BIN} PRIVATE ${GGJSON} ${GGJSONPB} ${GGSCRIBES} ${CURL} ${JSON})
  endif()
  target_link_libraries(${VS_VAHANA_BIN} PRIVATE ${OVR} ${OPENVR})
endif(WINDOWS)
set_target_properties(${VS_VAHANA_BIN} PROPERTIES COMPILE_FLAGS "${COMMON_FLAGS}")
add_cppcheck(${VS_VAHANA_BIN} VS)

