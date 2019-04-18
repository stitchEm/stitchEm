#!/bin/bash

set -e

MODE="${1-release}"

source make_dmg_functions.sh

# binary built by qmake
TARGET="videostitch-studio"

APP_SOURCE_DIR="videostitch-studio-gui"

# final name of the app
APPLICATION_BUNDLE="VideoStitch Studio"

DMG_NAME="VideoStitch-Studio"
APP=$TARGET
VERSION=$(../installer/genappsVersion.sh Studio)
initialize
TRANSLATIONS="${CMAKE_BUILD_DIR}/apps/src/${APP_SOURCE_DIR}"

PLUGINS_LIB="libjpgPlugin.dylib "
PLUGINS_LIB+="libpamPlugin.dylib "
PLUGINS_LIB+="libpngPlugin.dylib "
PLUGINS_LIB+="librawPlugin.dylib "
PLUGINS_LIB+="libtiffPlugin.dylib "
PLUGINS_LIB+="libavPlugin.dylib "

# put batchstitcher in root bin dir with the other binaries
"${QT_PATH}/macdeployqt" "${BINARIES}/batchstitcher.app"
cp $BINARIES/batchstitcher.app/Contents/MacOS/batchstitcher $BINARIES/batchstitcher



APPS=($BINARIES/calibrationimport $BINARIES/videostitch-cmd $APPLICATION_DIR/Videostitch-Studio $BINARIES/batchstitcher $BINARIES/ptvb2ptv)
package_apps
package_qt
package_plugins

make_dmg
