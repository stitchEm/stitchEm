#!/bin/bash -eu

MODE="${1-"release"}"
BUILD_FILES="../../bin/x64/${MODE}"
CORE_PLUGINS="${BUILD_FILES}/core_plugins"
EXTRA_PLUGINS="${BUILD_FILES}/vahana_plugins"

NAME="stitchingbox"
PREFIX="/opt/videostitch"
DIR="${NAME}${PREFIX}"
LIB="${DIR}/lib"
BIN="${DIR}/bin"
PROJECTS="${DIR}/projects"
PLUGINS="${LIB}/plugins"
LIB_ORIGIN="vs"
BIN_ORIGIN="samples"
EXTERNAL_DEPS="../../external_deps/lib"

#Package name
ARCHIVE_NAME="stitchingbox.deb"

#A server app
SERVER_APP="${BIN_ORIGIN}/server"

#Network service
NETWORK_SERVICE="${BIN_ORIGIN}/network"

#Dependencies

#Create folders
rm -rf "${DIR}"
mkdir -p "${LIB}"
mkdir -p "${BIN}"
mkdir -p "${PROJECTS}"
mkdir -p "${PLUGINS}"

# Version string with upstream_version as current UTC date
# debian_revision 0 to allow '-' in version number
echo "Version: 3.0-$(date -u +%Y%m%d-%H%M)-0" >> "${NAME}/DEBIAN/control"

#Copy libs
cp "${LIB_ORIGIN}/__init__.py" "${LIB}"
cp "${LIB_ORIGIN}/vs.py" "${LIB}"
cp "${LIB_ORIGIN}/_vs.so" "${LIB}"
cp "${LIB_ORIGIN}/libvideostitch.so" "${LIB}"

#Copy external deps
cp "${EXTERNAL_DEPS}/libopencv_core.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_calib3d.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_features2d.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_flann.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_imgproc.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_ml.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libopencv_video.so.3.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libturbojpeg.so.0" "${LIB}"
cp "${EXTERNAL_DEPS}/libpng16.so.16" "${LIB}"
cp "${EXTERNAL_DEPS}/libavcodec.so.57" "${LIB}"
cp "${EXTERNAL_DEPS}/libavformat.so.57" "${LIB}"
cp "${EXTERNAL_DEPS}/libavutil.so.55" "${LIB}"
cp "${EXTERNAL_DEPS}/libceres.so.1" "${LIB}"

#Copy plugins
cp "${CORE_PLUGINS}/libavPlugin.so" "${PLUGINS}"
cp "${CORE_PLUGINS}/libjpgPlugin.so" "${PLUGINS}"
cp "${CORE_PLUGINS}/libpamPlugin.so" "${PLUGINS}"
cp "${CORE_PLUGINS}/libpngPlugin.so" "${PLUGINS}"
cp "${CORE_PLUGINS}/librawPlugin.so" "${PLUGINS}"
cp "${CORE_PLUGINS}/libtiffPlugin.so" "${PLUGINS}"
cp "${EXTRA_PLUGINS}/librtmpPlugin.so" "${PLUGINS}"
cp "${EXTRA_PLUGINS}/libportaudioPlugin.so" "${PLUGINS}"

#Copy webapps
cp -a "${SERVER_APP}" "${BIN}"

#Copy network service
cp -a "${NETWORK_SERVICE}" "${BIN}"

#loads version file (a .ini like file containing definitions for apiVersion and serverVersion)
source <(grep = ./samples/server/version)

chmod -R 755 "${NAME}"
dpkg-deb --build "${NAME}"
mv "${ARCHIVE_NAME}" "${NAME}_${MODE}_${serverVersion}.deb"
