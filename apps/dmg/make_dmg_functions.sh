#!/bin/bash

function initialize {
  CMAKE_BUILD_DIR="../../cmake_build"
  BINARIES="${CMAKE_BUILD_DIR}/bin/x64/${MODE}"
  IO_BINARIES="${BINARIES}/CorePlugins"
  PACKAGE="${BINARIES}/${TARGET}.app"
  PLUGINS="${PACKAGE}/Contents/CorePlugins"
  APPLICATION_DIR="${PACKAGE}/Contents/MacOS"
  FRAMEWORKS="${PACKAGE}/Contents/Frameworks"
  OUTPUTDMG="dmg/${DMG_NAME}.dmg"
  LIBS="libvideostitch.dylib calibrationimport ptvb2ptv videostitch-cmd"
  EXT_DEPS="../external_deps"

  # work in the root directory
  cd ..
  QT_PATH="${QT_INSTALL}/$(cat qt.version)/clang_64/bin"
}

function package {
  # strip the symbols from the lib
  strip -x -u -r -arch all $1
  # for every non-system dependency
  local dylib
  for dylib in `otool -L $1 | sed '1d' | egrep -v "/usr/lib/|/System/|Qt|CUDA.framework" | awk '{print $1}'` ; do
    echo "$1 : dependency ${dylib}"
    # find its location
    fullname="${dylib}"
    fullname="${fullname/@rpath\/libHalf/${EXT_DEPS}/lib/openexr/libHalf}"
    fullname="${fullname/@rpath\/libIex/${EXT_DEPS}/lib/openexr/libIex}"
    fullname="${fullname/@rpath\/libIlm/${EXT_DEPS}/lib/openexr/libIlm}"
    fullname="${fullname/@rpath\/libImath/${EXT_DEPS}/lib/openexr/libImath}"
    fullname="${fullname/@rpath\/libglfw/${EXT_DEPS}/lib/glfw/libglfw}"
    fullname="${fullname/@rpath\/libceres/${EXT_DEPS}/lib/ceres/libceres}"
    fullname="${fullname/@rpath\/libopencv/${EXT_DEPS}/lib/opencv2/lib/libopencv}"
    fullname="${fullname/@rpath\/libcu//usr/local/cuda-8.0/lib/libcu}"
    fullname="${fullname/@rpath\/libturbojpeg/${EXT_DEPS}/lib/libjpeg-turbo/libturbojpeg}"
    fullname="${fullname/@rpath\/libjpeg/${EXT_DEPS}/lib/libjpeg-turbo/libjpeg}"
    fullname="${fullname/@rpath\/libav/${EXT_DEPS}/lib/ffmpeg/libav}"
    fullname="${fullname/@rpath\/libx264/${EXT_DEPS}/lib/ffmpeg/libx264}"
    fullname="${fullname/@rpath\/libmp3/${EXT_DEPS}/lib/ffmpeg/libmp3}"
    fullname="${fullname/@rpath\/libclang_rt.tsan_osx_dynamic.dylib//Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/8.0.0/lib/darwin/libclang_rt.tsan_osx_dynamic.dylib}"
    fullname="${fullname/@rpath/.//}"
    fullname="${fullname/@loader_path\/..//Applications/VLC.app/Contents/MacOS}"
    fullname="${fullname/@loader_path//Applications/VLC.app/Contents/MacOS}"
    fullname="${fullname/@executable_path\/..\/Frameworks/$BINARIES}"
    basefile="$(basename ${fullname})"
    # package it if needed
    if [ ! -f "${FRAMEWORKS}/${basefile}" ] && [ ! -f "${PLUGINS}/${basefile}" ] ; then
      echo "packaging ${basefile}"
      #Substitution needed because libvideostitch- have id without full path
      cp "${fullname/libvideostitch/$BINARIES/libvideostitch}" "${FRAMEWORKS}"
      chmod u+rwx "${FRAMEWORKS}/${basefile}"
      install_name_tool -id "@executable_path/../Frameworks/${basefile}" "${FRAMEWORKS}/${basefile}"

      # recursively package its non-system dependencies
      package "${FRAMEWORKS}/${basefile}"
    else
      echo "${basefile} has already been packaged"
    fi
    # the dynamic shared library is now in Frameworks
    echo "install_name_tool -change ${dylib} @executable_path/../Frameworks/$(basename ${dylib}) $1"
    install_name_tool -change "${dylib}" "@executable_path/../Frameworks/$(basename ${dylib})" "$1"
  done
  regex="path (.+) \(offset [0-9]+\)"
  otool -l "${1}" | grep "path /" | while read -r line ; do
    if [[ ${line} =~ ${regex} ]]; then
      path="${BASH_REMATCH[1]}"
      install_name_tool -delete_rpath "${path}" "${1}"
    else
      echo "something went wrong"
      echo "${line}"
      exit 1
    fi
  done
}

function package_apps {
  if [ ! -e "${PACKAGE}" ] ; then
    echo "Fatal error: the ${PACKAGE} doesn't exist!"
    exit 1
  fi
  if [ ! -e "${FRAMEWORKS}" ] ; then
    mkdir "${FRAMEWORKS}"
  fi

  # do what we can't do on Linux: change the dependency paths

  for elt in ${APPS[@]} ; do
    if [ -e ${elt} ] ; then
      if [[ "${elt}" != *"${APPLICATION_DIR}"* ]] ; then
        tmp="${elt}"
        elt="${elt/$BINARIES/$APPLICATION_DIR}"
        cp "${tmp}" "${elt}"
      fi
      package "${elt}"
    else
      echo "Warning : ${elt} has not been built"
    fi
  done

  LIBCUDA="libvideostitch_cuda.dylib"
  LIBOPENCL="libvideostitch_opencl.dylib"
  if [ -f "${BINARIES}/${LIBCUDA}" ] ; then
    cp "${BINARIES}/${LIBCUDA}" "${FRAMEWORKS}"
    package "${FRAMEWORKS}/${LIBCUDA}"
  fi
  if [ -f "${BINARIES}/${LIBOPENCL}" ] ; then
    cp "${BINARIES}/${LIBOPENCL}" "${FRAMEWORKS}"
    package "${FRAMEWORKS}/${LIBOPENCL}"
  fi

  libvideostitch="${FRAMEWORKS}/libvideostitch.dylib"
  libvideostitch_cuda="${FRAMEWORKS}/${LIBCUDA}"
  libvideostitch_opencl="${FRAMEWORKS}/${LIBOPENCL}"

  if [ ! -f "${libvideostitch_cuda}" ] && [ ! -f "${libvideostitch_opencl}" ] ; then
    echo "Fatal error: no libvideostitch backend found!"
    exit 1
  fi

  if [ -f "${libvideostitch}" ] ; then
    rm -f "${libvideostitch}"
  fi

  if [  -f "${libvideostitch_opencl}" ] ; then
    ln -s "libvideostitch_opencl.dylib" "${libvideostitch}"
  else
    ln -s "libvideostitch_cuda.dylib" "${libvideostitch}"
  fi
}

function package_qt {
  if [ -f "${PACKAGE}/Contents/MacOs/VideoStitch-Studio" ]; then
    mv "${PACKAGE}/Contents/MacOs/VideoStitch-Studio" "${PACKAGE}/Contents/MacOs/VideoStitch Studio"
  fi

  "${QT_PATH}/macdeployqt" "${PACKAGE}"

  # fix QT Frameworks path
  elt="${APPLICATION_DIR}/Videostitch Studio"
  echo "QT for ${elt}"
  for qtlib in $(otool -L "${elt}" | grep Qt | awk '{print $1}'); do
    echo "installing ${qtlib}"
    install_name_tool -change "${qtlib}" "${qtlib/@rpath/@executable_path/../Frameworks}" "${elt}"
    lib="$(dirname "${elt}")/${qtlib/@rpath/../Frameworks}"
    install_name_tool -id "${qtlib/@rpath/@executable_path/../Frameworks}" "${lib}"
    for qtlib2 in $(otool -L ${lib} | grep @rpath | grep Qt | awk '{print $1}'); do
      install_name_tool -change "${qtlib2}" "${qtlib2/@rpath/@executable_path/../Frameworks}" "${lib}"
    done
  done
  for vs_lib in ${APPLICATION_DIR}/../Frameworks/libvideostitch-base.dylib ${APPLICATION_DIR}/../Frameworks/libvideostitch-gui.dylib ${APPLICATION_DIR}/batchstitcher; do
    echo "QT for ${vs_lib}"
    for qtlib in $(otool -L "${vs_lib}" | grep Qt | awk '{print $1}'); do
      install_name_tool -change "${qtlib}" "${qtlib/@rpath/@executable_path/../Frameworks}" "${vs_lib}"
    done
  done
  for qt_plugin in ${APPLICATION_DIR}/../PlugIns/*/*.dylib; do
    echo "changing ${qt_plugin}"
    for qtlib in $(otool -L "${qt_plugin}" | grep Qt | awk '{print $1}'); do
      install_name_tool -change "${qtlib}" "${qtlib/@rpath/@executable_path/../Frameworks}" "${qt_plugin}"

    done
  done
}

function package_plugins {

  if [ ! -e "${PACKAGE}" ] ; then
    echo "Fatal error: the ${PACKAGE} doesn't exist!"
    exit 1
  fi
  if [ ! -e "${FRAMEWORKS}" ] ; then
    mkdir "${FRAMEWORKS}"
  fi
  if [ ! -e "${PLUGINS}" ] ; then
    mkdir "${PLUGINS}"
  fi

  for plugin in ${PLUGINS_LIB}; do
    basefile=$(basename ${plugin})
    echo "packaging plugin: ${basefile}"

    # this is the code from package() with different arguments, but I feel introducing new variables
    # would make it harder to understand what's actually going on for future readers
    cp "${IO_BINARIES}/${plugin}" "${PLUGINS}"
    chmod u+rwx "${PLUGINS}/${basefile}"
    install_name_tool -id "@executable_path/../CorePlugins/${basefile}" "${PLUGINS}/${basefile}"

    # recursively package its non-system dependencies
    package "${PLUGINS}/${basefile}"
  done
}

function make_dmg {
  #handle meta + version
  echo "version is ${VERSION}"
  gsed 's/<string>9.9.9<\/string>/<string>'"${VERSION}"'<\/string>/' "dmg/${APP}.plist" > "${PACKAGE}/Contents/Info.plist"

  # temporary fix for umount error: kill squish binaries accessing the disk image
  pkill videostitch-studio || true

  # $TARGET makes it easier to handle version numbers in the future
  if [ -d "/Volumes/${TARGET}" ]; then
    hdiutil detach -verbose "/Volumes/${TARGET}"
  fi

  # give all users read and execute access to the .app, recursively
  chmod -R og+rx "${PACKAGE}"
  # codesign --deep --force --verify --verbose=4 --sign "VIDEOSTITCH" "${PACKAGE}"
  # codesign --deep --verify --verbose=4 "${PACKAGE}"

  TMP="dmg/videostitch_tmp.dmg"
  hdiutil create "${TMP}" -volname "${TARGET}" -srcfolder "${PACKAGE}" -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -ov

  #we need to mount the dmg to set the alias to /Applications
  #and sleep because of MacOS bullshitness
  sleep 2
  device=$(hdiutil attach -readwrite -noverify -noautoopen "${TMP}" | egrep '^/dev/' | sed 1q | awk '{print $1}')
  trap "hdiutil detach $device; exit" INT TERM EXIT

  sleep 5
  echo "Device: $device"

  DMG_MOUNT="/Volumes/${TARGET}"

  # give the package a name with spaces
  # would need to change nearly every line of this script to escape spaces if done from beginning
  mv "${DMG_MOUNT}/${TARGET}.app" "${DMG_MOUNT}/${APPLICATION_BUNDLE}.app"

  #copy the background image in
  mkdir "${DMG_MOUNT}/.background"
  cp "dmg/logo_background.png" "${DMG_MOUNT}/.background/"

  #copy the link to the nVidia website
  cp "dmg/List of supported graphics cards.webloc" "${DMG_MOUNT}/List of supported graphics cards.webloc"

  #create an alias to Applications
  ln -sf /Applications "${DMG_MOUNT}/Applications"

  #dmg window dimensions
  dmg_width=522
  dmg_height=445
  dmg_topleft_x=100
  dmg_topleft_y=100
  dmg_bottomright_x=$(expr ${dmg_topleft_x} + ${dmg_width})
  dmg_bottomright_y=$(expr ${dmg_topleft_y} + ${dmg_height})

  #set the background image and icon location
  echo '
     tell application "Finder"
       tell disk "'${TARGET}'"
             open
             tell container window
               set toolbar visible to false
               set statusbar visible to false
               set current view to icon view
               delay 1 -- sync
               set the bounds to {'${dmg_topleft_x}', '${dmg_topleft_y}', '${dmg_bottomright_x}', '${dmg_bottomright_y}'}
             end tell
             delay 1 -- sync
             set theViewOptions to the icon view options of container window
             set arrangement of theViewOptions to not arranged
             set icon size of theViewOptions to 104
             set background picture of theViewOptions to file ".background:'logo_background.png'"
             set position of item "'${APPLICATION_BUNDLE}.app'" of container window to {120, 125}
             set position of item "'Applications'" of container window to {385, 125}
             set position of item "'List of supported graphics cards.webloc'" of container window to {260, 300}
             close
             open
             update without registering applications
             delay 5
       end tell
     end tell
  ' | osascript

  trap - INT TERM EXIT

  echo "Detaching device ${device}"
  hdiutil detach ${device}

  #convert to compressed image, delete temp image
  hdiutil convert "${TMP}" -format UDZO -imagekey zlib-level=9 -o "${OUTPUTDMG}" -ov
  if [ -f "${TMP}" ] ; then
    rm -f "${TMP}"
  fi

  chmod og+rx "${OUTPUTDMG}"
}
