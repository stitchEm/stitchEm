#!/bin/bash -eu

MODE="${1-release}"
BINARIES="../../bin/x64/${MODE}"
IO_BINARIES="${BINARIES}/core_plugins"
VAH_IO_BINARIES="${BINARIES}/vahana_plugins"
EXTERNAL_BINARIES="../../external_deps/lib"
PRODUCT=VahanaVR
CMD=videostitch-cmd
BIN="${PRODUCT}/bin/"
LIB="${PRODUCT}/lib/"
PLUGINS="${BIN}/core_plugins/"
VAH_PLUGINS="${BIN}/vahana_plugins/"
CALIBRATION=calibrationimport
LIST_CO_PLUGINS="libjpgPlugin.so libavPlugin.so libpamPlugin.so libpngPlugin.so "
LIST_CO_PLUGINS+="librawPlugin.so libtiffPlugin.so"
LIST_VA_PLUGINS="librtmpPlugin.so"
VCUDA=$(cat ../../cuda.version)
CUDA_PATH="/usr/local/cuda-${VCUDA}/lib64"

rm -rf "${PRODUCT}"
mkdir "${PRODUCT}"
mkdir "${BIN}"
mkdir "${LIB}"
mkdir "${PLUGINS}"
mkdir "${VAH_PLUGINS}"

cp README "${PRODUCT}/"
cp app_template "${PRODUCT}/${CALIBRATION}"
cp app_template "${PRODUCT}/${CMD}"
cp app_template "${PRODUCT}/${PRODUCT}"

cp launcher "${BIN}"
cp "${BINARIES}/${CALIBRATION}" "${BIN}"
cp "${BINARIES}/${CMD}" "${BIN}"
cp "${BINARIES}/${PRODUCT}" "${BIN}"

for plugin in ${LIST_CO_PLUGINS}; do
    cp "${IO_BINARIES}/${plugin}" "${PLUGINS}"
done
for plugin in ${LIST_VA_PLUGINS}; do
    cp "${VAH_IO_BINARIES}/${plugin}" "${PLUGINS}"
done

cp "${BINARIES}/libvideostitch_cuda.so" "${LIB}"
cp "${BINARIES}/libvideostitch-gui"* "${LIB}"
cp "${BINARIES}/libvideostitch-base"* "${LIB}"
cp "${BINARIES}/libvideostitch-gpudiscovery"* "${LIB}"

cp "${EXTERNAL_BINARIES}/libpng16.so.16" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_core.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_calib3d.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_ml.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_features2d.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_flann.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_imgproc.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libopencv_video.so.3.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libturbojpeg.so.0" "${LIB}"
cp "${EXTERNAL_BINARIES}/libmfxhw64.so" "${LIB}"
# we should not distribute the dependencies manually
cp "${EXTERNAL_BINARIES}/libavcodec.so.57" "${LIB}"
cp "${EXTERNAL_BINARIES}/libavformat.so.57" "${LIB}"
cp "${EXTERNAL_BINARIES}/libavutil.so.55" "${LIB}"
cp "${EXTERNAL_BINARIES}/libva.so.1" "${LIB}"
cp "${EXTERNAL_BINARIES}/libx264.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libceres.so.1" "${LIB}"

cp "${EXTERNAL_BINARIES}/libglog.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libglog.so.0" "${LIB}"

cp "${EXTERNAL_BINARIES}/libcurl.so.4" "${LIB}"
cp "${EXTERNAL_BINARIES}/libcurl.so" "${LIB}"

cp "${EXTERNAL_BINARIES}/libssl.so.1.0.0" "${LIB}"
cp "${EXTERNAL_BINARIES}/libssl.so" "${LIB}"

cp "${EXTERNAL_BINARIES}/libjsoncpp.so" "${LIB}"

cp "${EXTERNAL_BINARIES}/libgoogleapis_internal.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_utils.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_oauth2.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_http.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_curl_http.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogle_youtube_api.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_jsoncpp.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_openssl_codec.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libgoogleapis_json.so" "${LIB}"

cp "${CUDA_PATH}/libcudart.so.${VCUDA}" "${LIB}"

chmod +x -R "${PRODUCT}/"
zip -r "VideoStitch-${PRODUCT}.zip" "${PRODUCT}/"
rm -rf "${PRODUCT}/"
