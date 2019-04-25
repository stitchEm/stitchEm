#
# Generates python bindings
# Copies files needed for the .deb file generation into the INSTALLATION_DIR directory
#


# ----------------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------------

include(UseSWIG)
include(FindPythonLibs)

if(NOT PYTHONLIBS_FOUND)
  message(FATAL_ERROR "Could not find python libs, unable to generate bindings")
else()
  message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
  message(STATUS "PYTHON_INCLUDE_PATH: ${PYTHON_INCLUDE_PATH}")
  message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
  message(STATUS "PYTHON_DEBUG_LIBRARIES: ${PYTHON_DEBUG_LIBRARIES}")
  message(STATUS "PYTHONLIBS_VERSION_STRING: ${PYTHONLIBS_VERSION_STRING}")
endif()

find_package(SWIG REQUIRED)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOW)

set (DEST_PKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/box_package")
file (MAKE_DIRECTORY ${DEST_PKG_DIR})

set(BASE_DIR "${CMAKE_SOURCE_DIR}/lib/bindings")
set(SRC_PKG_DIR "${BASE_DIR}/stitchingbox")

set(INSTALLATION_DIR "${DEST_PKG_DIR}/stitchingbox")

set(OPT_DIR "${INSTALLATION_DIR}/opt/videostitch")
set(OPT_LIB_DIR "${OPT_DIR}/lib/${CMAKE_BUILD_TYPE_LOW}/")
set(OPT_PLUGINS_DIR "${OPT_LIB_DIR}/plugins")
set(OPT_BIN_DIR "${OPT_DIR}/bin")
set(OPT_PROJECTS_DIR "${OPT_DIR}/projects")

set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/lib/bindings/vs")

set(SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/lib/bindings/samples")
set(SERVER_SCRIPTS_DIR "${SCRIPTS_DIR}/server")
set(FIRMWARE_DIR "${SERVER_SCRIPTS_DIR}/firmware")

set (VERSION_FILE "${SERVER_SCRIPTS_DIR}/API/version")

set (DOCS_DIR "${SERVER_SCRIPTS_DIR}/static/docs")

# ----------------------------------------------------------------------------
# Customs target to manage the package
# ----------------------------------------------------------------------------


file(WRITE ${DEST_PKG_DIR}/check_firmware.py
"
import hashlib
import ConfigParser
import json
import sys
from os import path as osp

config = ConfigParser.SafeConfigParser()
config.readfp(open(sys.argv[1]))
updates = json.loads(config.get('DEFAULT', 'firmwareUpdates'))
for update in updates:
  file_path = osp.join(sys.argv[2], update.get('file') + '.fwupd')
  with open(file_path, 'rb') as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    if md5.lower() != update.get('md5').lower():
      exit(-1)

exit(0)
"
  )

file (WRITE ${DEST_PKG_DIR}/check_firmware.sh
 "python ${DEST_PKG_DIR}/check_firmware.py ${VERSION_FILE} ${SERVER_SCRIPTS_DIR}/firmware/
 ")

add_custom_target(check-firmware
                  COMMAND chmod 755 ./check_firmware.sh
                  COMMAND ./check_firmware.sh
                  WORKING_DIRECTORY ${DEST_PKG_DIR})

#Clean lib directory with double
file (WRITE ${DEST_PKG_DIR}/img_clean.sh
 "#!/bin/bash

  strip ${OPT_LIB_DIR}/_vs.so
  strip ${OPT_LIB_DIR}/plugins/*.so
  strip ${OPT_LIB_DIR}/*.so*

  rm -rf ${OPT_BIN_DIR}/server/test/
  rm -rf ${OPT_BIN_DIR}/server/debug/

  for i in ${OPT_LIB_DIR}/*.so.*[0-9].*[0-9]
  do
     bname=$(basename $i)
     cbname=\${bname\%\%.\*}
     dir=$(dirname $i)
     rm $dir/$cbname.so
     rm $dir/$cbname.so.[0-9]
     rm $dir/$cbname.so.[0-9][0-9]

     ln -s ./$bname $dir/$cbname.so
     ext1=\${bname\#\*.so.}
     ext2=\${ext1\%\%.\*}
     ln -s ./$bname $dir/$cbname.so.$ext2
  done
 ")




# Create shell script to package the version (prevents 'escaping hell' when trying to do that from COMMAND)
file (WRITE ${DEST_PKG_DIR}/dpkg.sh
 "version=$(awk -F '=' '/serverVersion=/ {print $2}' ${VERSION_FILE});
  api=$(awk -F '=' '/apiVersion=/ {print $2}' ${VERSION_FILE});
  now=$(date +'%d-%m-%Y %H:%M');
  rev=$(git --git-dir=${CMAKE_SOURCE_DIR}/.git name-rev HEAD --name-only);
  sha1=$(git --git-dir=${CMAKE_SOURCE_DIR}/.git rev-parse HEAD);
  build_branch=$(echo \${rev}-\${sha1});
  echo Packaging:
  echo SERVER \${version}
  echo BUILD BRANCH $build_branch
  echo CONFIG ${CMAKE_BUILD_TYPE_LOW}
  cp ${SRC_PKG_DIR}/DEBIAN/control ${INSTALLATION_DIR}/DEBIAN/
  echo Version:  3.0-\$(date -u +%Y%m%d-%H%M)-0 >> ${INSTALLATION_DIR}/DEBIAN/control
  echo \${version}\\(\${api}\\) \${build_branch} ${CMAKE_BUILD_TYPE_LOW} \${now} > ${OPT_BIN_DIR}/server/BuildVersionFile
  touch stitchingbox/opt/videostitch/${CMAKE_BUILD_TYPE_LOW}-SERVER_\${version}-API_\${api}-\${build_branch}
  chmod -R 755 ${INSTALLATION_DIR}/DEBIAN
  dpkg-deb -z1 --build stitchingbox stitchingbox-${CMAKE_BUILD_TYPE_LOW}-SERVER_\${version}-API_\${api}-\${build_branch}.deb
 ")

add_custom_target(create-box-package
                  DEPENDS check-firmware
                  COMMAND chmod 755 ./dpkg.sh
                  COMMAND ./dpkg.sh
                  WORKING_DIRECTORY ${DEST_PKG_DIR})

add_custom_target(get-box-package-dir
                  COMMAND echo ${OPT_LIB_DIR})


# Create shell script to package the stitchingbox firmware for buildroot img
file (WRITE ${DEST_PKG_DIR}/sb_fw_dev.sh
 "version=$(awk -F '=' '/serverVersion=/ {print $2}' ${VERSION_FILE});
  api=$(awk -F '=' '/apiVersion=/ {print $2}' ${VERSION_FILE});
  now=$(date +'%d-%m-%Y %H:%M');
  rev=$(git --git-dir=${CMAKE_SOURCE_DIR}/.git name-rev HEAD --name-only);
  sha1=$(git --git-dir=${CMAKE_SOURCE_DIR}/.git rev-parse HEAD);
  build_branch=$(echo \${rev}-\${sha1});
  echo SB FW building:
  echo SERVER \${version}
  echo BUILD BRANCH $build_branch
  echo CONFIG ${CMAKE_BUILD_TYPE_LOW}
  echo \${version}\\(\${api}\\) \${build_branch} ${CMAKE_BUILD_TYPE_LOW} \${now} > ${OPT_BIN_DIR}/server/BuildVersionFile
  touch stitchingbox/opt/videostitch/${CMAKE_BUILD_TYPE_LOW}-SERVER_\${version}-API_\${api}-\${build_branch}
  tar cvjf Orah_dev.img -C ${INSTALLATION_DIR}/opt/videostitch/ . --owner=1042 --group=1042
 ")

file (WRITE ${DEST_PKG_DIR}/sb_fw_prod.sh
  "tar cvjf Orah_prod.img -C ${INSTALLATION_DIR}/opt/videostitch/ . --owner=1042 --group=1042
 ")


add_custom_target(create-sb-fw-img
                  DEPENDS check-firmware
                  COMMAND chmod 755 ./sb_fw_dev.sh
                  COMMAND ./sb_fw_dev.sh
                  COMMAND chmod 755 ./img_clean.sh
                  COMMAND ./img_clean.sh
                  COMMAND chmod 755 ./sb_fw_prod.sh
                  COMMAND ./sb_fw_prod.sh
                  WORKING_DIRECTORY ${DEST_PKG_DIR})


# ----------------------------------------------------------------------------
# Build python bindings
#     generates: CMAKE_LIBRARY_OUTPUT_DIRECTORY/_vs.so, CMAKE_SWIG_OUTDIR/vs.py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Create docstrings from doxygen
# ----------------------------------------------------------------------------
set(SWIG_DOXYGEN ${BINDINGS_DIR}/vs_doxygen.i)

set(LIB_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../doc)
set(LIB_DOC_XML ${LIB_DOC_DIR}/xml/index.xml)

# Create Doxygen XML file
add_custom_command(OUTPUT ${LIB_DOC_XML}
                   COMMAND ${DOXYGEN_EXECUTABLE}
                   ARGS "${LIB_DOC_DIR}/Doxyfile-library-xml"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
                   )

# Use doxy2swig to create vs_doxygen.i
add_custom_command(OUTPUT ${SWIG_DOXYGEN}
                   COMMAND ${LIB_DOC_DIR}/doxy2swig/doxy2swig.py
                   ARGS ${LIB_DOC_XML} ${SWIG_DOXYGEN}
                   DEPENDS ${LIB_DOC_XML}
                   )
# ----------------------------------------------------------------------------

set(CMAKE_SWIG_FLAGS "${SWIG_FLAGS}" "-modern" "-builtin")

if(APPLE)
  # using MacPorts glfw for now
  find_library(glfw glfw HINTS "/opt/local/lib" REQUIRED NO_DEFAULT_PATH)
elseif(LINUX)
  find_library(glfw glfw REQUIRED)
else()
  find_library(glfw glfw HINTS "${CMAKE_EXTERNAL_LIB}" REQUIRED NO_DEFAULT_PATH)
endif()

set(CMAKE_SWIG_OUTDIR "${DEST_PKG_DIR}/bindings")
set(BINDING_SOURCE "vs/videostitch.i")
include_directories("${PYTHON_INCLUDE_DIRS}" "${BINDINGS_DIR}")
# include lib headers from lib/src
include_directories(${VS_DISCOVERY_PUBLIC_HEADERS_DIR})
include_directories(${VS_LIB_PUBLIC_HEADERS_DIR})
if(APPLE)
  include_directories(/opt/local/include)
endif()
set_property(SOURCE "${BINDING_SOURCE}" PROPERTY CPLUSPLUS ON)
set(SWIG_MODULE_vs_EXTRA_DEPS "vs/runtime.i" "vs/extensions.i" "vs/definitions.i" "vs/algorithmListenerGIL.hpp" "vs/CppCallback.hpp" "vs/renderer.hpp" "vs/compositor.hpp" "vs/stitch_loop.hpp" ${SWIG_DOXYGEN} ${VS_LIB})
swig_add_module(vs python "${BINDING_SOURCE}")
swig_link_libraries(vs ${glfw} ${PYTHON_LIBRARIES})
if(APPLE)
  swig_link_libraries(vs ${VS_LIB_FAKE})
else()
  swig_link_libraries(vs ${VS_LIB_DEFAULT})
endif()
set_source_files_properties(${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-std=c++14 -w")
set_source_files_properties(${swig_generated_file_fullname} PROPERTIES OBJECT_DEPENDS ${SWIG_DOXYGEN})

# ----------------------------------------------------------------------------
# Install base package hierearchy
# ----------------------------------------------------------------------------

install(
  DIRECTORY
    "${SRC_PKG_DIR}/"
  USE_SOURCE_PERMISSIONS
  DESTINATION
    "${INSTALLATION_DIR}"
  )

# ----------------------------------------------------------------------------
# Install plugins
# ----------------------------------------------------------------------------

install(
  DIRECTORY
    "${VS_PLUGIN_DIR}/"
    "${VS_VAHANA_PLUGIN_DIR}/"
  DESTINATION
    "${OPT_PLUGINS_DIR}"
  )

# ----------------------------------------------------------------------------
# Install external libs
#   TODO : We are probably installing too much. We should find a way to do some cleanup
# ----------------------------------------------------------------------------

install(
  DIRECTORY
    "${CMAKE_EXTERNAL_LIB}/"
  DESTINATION
    "${OPT_LIB_DIR}"
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN "*.so*"
  PATTERN "iHD*" EXCLUDE
  PATTERN "libva*" EXCLUDE
  PATTERN "libmfx*" EXCLUDE
  PATTERN "libdrm*" EXCLUDE
)

file (GLOB GLEWS "${GLEW_LIBRARIES}*")
install(
  FILES
    # Quickfix : Make sure that we package the right libglew version
    ${GLEWS}
  DESTINATION
    "${OPT_LIB_DIR}"
  )

# ----------------------------------------------------------------------------
# Install bindings
# ----------------------------------------------------------------------------

install(
  FILES
    "${BINDINGS_DIR}/__init__.py"
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_vs.so"
    "${CMAKE_SWIG_OUTDIR}/vs.py"
  DESTINATION
    "${OPT_LIB_DIR}"
  )

# ----------------------------------------------------------------------------
# Install firmware
# ----------------------------------------------------------------------------

file(MAKE_DIRECTORY "${FIRMWARE_DIR}")

install(
  DIRECTORY
    "${CMAKE_EXTERNAL_LIB}/"
  DESTINATION
    "${FIRMWARE_DIR}"
  FILES_MATCHING PATTERN "*.fwupd"
  PATTERN "pkgconfig" EXCLUDE
  )


# ----------------------------------------------------------------------------
# Install docs
# ----------------------------------------------------------------------------

install(
  DIRECTORY
    "${CMAKE_EXTERNAL_LIB}/sb-docs/"
  DESTINATION
    "${DOCS_DIR}"
  FILES_MATCHING PATTERN "*.pdf"
  )


# ----------------------------------------------------------------------------
# Install python scripts and server stuff
# ----------------------------------------------------------------------------

install(
  DIRECTORY
    "${SERVER_SCRIPTS_DIR}"
  USE_SOURCE_PERMISSIONS
  DESTINATION
    "${OPT_BIN_DIR}"
  PATTERN
    "*.pyc" EXCLUDE
  )


# ----------------------------------------------------------------------------
# Install libvideostitch
# ----------------------------------------------------------------------------

install(
  FILES
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${VS_LIB_DEFAULT}.so"
  DESTINATION
    "${OPT_LIB_DIR}"
  )

# ----------------------------------------------------------------------------
# Install libvideostitch-gpudiscovery
# ----------------------------------------------------------------------------

install(
  FILES
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libvideostitch-gpudiscovery.so"
  DESTINATION
    "${OPT_LIB_DIR}"
  )

