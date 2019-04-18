#!/bin/bash -e

if [ "${1}" == "" ]; then
    MODE="release"
else
    MODE="${1}"
fi
BINARIES="../../bin/x64/${MODE}"
IO_BINARIES="${BINARIES}/core_plugins"
EXTERNAL_BINARIES="../../external_deps/lib"
CORE_PLUGIN="core_plugins"
VCUDA=$(cat ../../cuda.version)
CUDA_PATH="/usr/local/cuda-${VCUDA}/lib64"
LIB="videostitch-studio/lib"

mkdir videostitch-studio
mkdir videostitch-studio/bin
mkdir "${LIB}"
mkdir videostitch-studio/bin/${CORE_PLUGIN}

cp README videostitch-studio
cp app_template videostitch-studio/calibrationimport
cp app_template videostitch-studio/videostitch-cmd
cp app_template videostitch-studio/videostitch-studio
cp app_template videostitch-studio/batchstitcher

cp launcher videostitch-studio/bin
cp $BINARIES/calibrationimport videostitch-studio/bin
cp $BINARIES/videostitch-cmd videostitch-studio/bin
cp $BINARIES/videostitch-studio videostitch-studio/bin/videostitch-studio
cp $BINARIES/batchstitcher videostitch-studio/bin

cp $BINARIES/libvideostitch_cuda.so "${LIB}"
cp $BINARIES/libvideostitch-gui* "${LIB}"
cp $BINARIES/libvideostitch-base* "${LIB}"
cp $BINARIES/libvideostitch-gpudiscovery* "${LIB}"

#adding plugins
cp $IO_BINARIES/libjpgPlugin.so videostitch-studio/bin/${CORE_PLUGIN}
cp $IO_BINARIES/libavPlugin.so videostitch-studio/bin/${CORE_PLUGIN}
cp $IO_BINARIES/libpamPlugin.so videostitch-studio/bin/${CORE_PLUGIN}
cp $IO_BINARIES/libpngPlugin.so videostitch-studio/bin/${CORE_PLUGIN}
cp $IO_BINARIES/librawPlugin.so videostitch-studio/bin/${CORE_PLUGIN}
cp $IO_BINARIES/libtiffPlugin.so videostitch-studio/bin/${CORE_PLUGIN}

cp $EXTERNAL_BINARIES/libturbojpeg.so.0 "${LIB}"
cp $EXTERNAL_BINARIES/libjpeg.so.62 "${LIB}"
cp $EXTERNAL_BINARIES/libpng16.so.16 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_core.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_calib3d.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_ml.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_features2d.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_flann.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_imgproc.so.3.1 "${LIB}"
cp $EXTERNAL_BINARIES/libopencv_video.so.3.1 "${LIB}"
# we should not distribute these dependencies manually
cp /usr/lib/x86_64-linux-gnu/libtiff.so.5 "${LIB}"
cp "${EXTERNAL_BINARIES}/libavcodec.so.57" "${LIB}"
cp "${EXTERNAL_BINARIES}/libavformat.so.57" "${LIB}"
cp "${EXTERNAL_BINARIES}/libavutil.so.55" "${LIB}"
cp "${EXTERNAL_BINARIES}/libx264.so" "${LIB}"
cp "${EXTERNAL_BINARIES}/libceres.so.1" "${LIB}"
cp "${CUDA_PATH}/libcudart.so.${VCUDA}" "${LIB}"
cp "/lib/x86_64-linux-gnu/libz.so.1" "${LIB}"

chmod +x -R videostitch-studio/
zip -r VideoStitch-Studio.zip videostitch-studio/
rm -rf videostitch-studio/
