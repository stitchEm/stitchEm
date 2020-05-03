project(VideoStitch_lib)

# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

if((VS_LIB_UNIT_TEST EQUAL VS_LIB_STATIC) AND (NOT BUILD_STATIC_LIB))
  message(FATAL_ERROR "Static libvideostitch (BUILD_STATIC_LIB) needed for unit tests")
endif()

option(BUILD_LIB_TESTS "Build libvideostitch unit tests" ON)

#-----------------------------------------------------------------------------
# Python bindings
#-----------------------------------------------------------------------------
option(GENERATE_BINDINGS "Generate libvideostitch Python bindings" OFF)

# ----------------------------------------------------------------------------
# The object lib that will be populated by CUDA and OpenCL backend CMake
# to be made into ${VS_LIB_CUDA}, ${VS_LIB_OPENCL} and ${VS_LIB_STATIC}
set(VS_LIB_OBJECTS_CUDA vslib-objects-cuda)
set(VS_LIB_OBJECTS_OPENCL vslib-objects-opencl)

# add libraries that VS_LIB_CUDA or VS_LIB_OPENCL will link to
# populated from the GPU backend CMake files
macro(vs_lib_link_libraries)
  set(ARGV_COPY ${ARGV})
  list(GET ARGV_COPY 0 CONF)
  list(REMOVE_AT ARGV_COPY 0)
  set(VS_LIB_${CONF}_LIBRARIES_TO_LINK
      ${VS_LIB_${CONF}_LIBRARIES_TO_LINK}
      ${ARGV_COPY})
endmacro()

macro(vs_lib_target_link_libraries TARGET_LIB)
  if(TEGRA_DEMO)
    # Some code in /usr/share/cmake-3.*/Modules/FindCUDA.cmake make use do that
    # "All uses of target_link_libraries with a target must be either all-keyword or all-plain."
    target_link_libraries(${TARGET_LIB}
                          ${VS_LIB_PUBLIC_${BACKEND_NAME}_LIBRARIES_TO_LINK} ${VS_LIB_PUBLIC_COMMON_LIBRARIES_TO_LINK}
                          ${VS_LIB_${BACKEND_NAME}_LIBRARIES_TO_LINK} ${VS_LIB_COMMON_LIBRARIES_TO_LINK})
  else()
    target_link_libraries(${TARGET_LIB}
                          PUBLIC ${VS_LIB_PUBLIC_${BACKEND_NAME}_LIBRARIES_TO_LINK} ${VS_LIB_PUBLIC_COMMON_LIBRARIES_TO_LINK}
                          PRIVATE ${VS_LIB_${BACKEND_NAME}_LIBRARIES_TO_LINK} ${VS_LIB_COMMON_LIBRARIES_TO_LINK})
  endif()
endmacro()


# ----------------------------------------------------------------------------
set(VS_LIB_SYSTEM_INCLUDES
    ${CMAKE_EXTERNAL_DEPS}/include/libpng
    ${CMAKE_EXTERNAL_DEPS}/include/glm
    ${CMAKE_EXTERNAL_DEPS}/include
)

find_package(Threads REQUIRED)


# ----------------------------------------------------------------------------
# OS specific libvideostitch settings
# TODO: to be removed
# -- Library/packages discovery should happen in os independent cmake scripts
# -- Flag settings should be included directly in the current file
# ----------------------------------------------------------------------------
if(APPLE)
  include(apple.cmake)
elseif(LINUX)
  include(linux.cmake)
elseif(WINDOWS)
  include(windows.cmake)
endif(APPLE)

if(TEGRA_DEMO)
  add_definitions(-DTEGRA_DEMO)
endif(TEGRA_DEMO)
if(ANDROID)
  add_definitions(-DGLFWLIB_UNSUPPORTED)
endif(ANDROID)

if (NOT COMMAND find_host_package)
  macro( find_host_package )
    find_package( ${ARGN} )
  endmacro()
endif(NOT COMMAND find_host_package)

find_host_package(BISON 3.0)
find_host_package(FLEX)

# ----------------------------------------------------------------------------
# Compiler flags for the lib
# ----------------------------------------------------------------------------

if(MSVC)
  set(LIB_FLAGS ${CMAKE_CXX_FLAGS})
  STRING(REPLACE "/W3" "/W4" LIB_FLAGS "${LIB_FLAGS}")

  # C4251: We have a lot of those: FIXME
  # C4505: This warning is triggered by external headers (dirent.h)
  # C4503: Length of symbols issue. Impacts at debug time
  # C4512: We have a lot of those: FIXME
  # C4611: Shouldn't impact us
  # C4661: A member of the template class is not defined: FIXME
  # C4800: We often use conversion from internal types to bool,
  # to fix this warning we should implement directly those function
  set(CMAKE_CXX_FLAGS
      "${LIB_FLAGS} /wd4251 /wd4503 /wd4505 /wd4512 /wd4611 /wd4661 /wd4800 /fp:precise")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} /GL")
elseif(COMPILER_CLANG)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers")
elseif(COMPILER_GCC)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-function -Wno-missing-field-initializers")
endif()

# ----------------------------------------------------------------------------
# Create JSON parser
# Created in sub-sub-folder for compatibility with scons build
# ----------------------------------------------------------------------------
set(JSON_PARSER_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/json)
set(JSON_PARSER_GEN_FILES_DIR ${JSON_PARSER_ROOT_DIR}/parser-generated)
file(MAKE_DIRECTORY ${JSON_PARSER_GEN_FILES_DIR})
bison_target(jsonParser src/parse/json.yy ${JSON_PARSER_GEN_FILES_DIR}/jsonParser.cpp)
flex_target(jsonLexer src/parse/json.ll  ${JSON_PARSER_GEN_FILES_DIR}/jsonLexer.cpp)
add_flex_bison_dependency(jsonLexer jsonParser)
# shut up warnings for generated file
if(WINDOWS)
  set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES CMAKE_CXX_FLAGS "-w")
elseif(APPLE)
  set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES COMPILE_FLAGS "-w -Xanalyzer -analyzer-disable-all-checks")
else()
  set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES COMPILE_FLAGS "-w")
endif(WINDOWS)
add_custom_target(jsonParserGeneration DEPENDS ${BISON_jsonParser_OUTPUTS})

# ----------------------------------------------------------------------------
# Generate header files containing the version number
# ----------------------------------------------------------------------------
add_custom_command(
    OUTPUT ${VERSION_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DVERSION_TEMPLATE="${VS_LIB_PUBLIC_HEADERS_DIR}/libvideostitch/version.hpp.templ"
        -DVERSION_HEADER=${VERSION_HEADER}
        -DVS_LIB_PUBLIC_HEADERS_DIR=${VS_LIB_PUBLIC_HEADERS_DIR}
        -DWORKING_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/libversion.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS ${VS_LIB_PUBLIC_HEADERS_DIR}/libvideostitch/version.hpp.templ
    )

# We don't need to do anything usefull here, it's just for
# libvideostitch-base to wait
if(WINDOWS)
  add_custom_target(
    CheckVersionHeader
    COMMAND dir
    DEPENDS ${VERSION_HEADER}
    )
else()
  add_custom_target(
    CheckVersionHeader
    COMMAND ls ${VERSION_HEADER}
    DEPENDS ${VERSION_HEADER}
    )
endif()

# ----------------------------------------------------------------------------
# Set libvideostitch sources
# ----------------------------------------------------------------------------
set(CORE_LIB_SOURCES
    ${HEADER_SOURCES}
    ${CORE_LIB_SOURCES}
    ${BISON_jsonParser_OUTPUTS}
    ${FLEX_jsonLexer_OUTPUTS}
    src/audio/ambisonic.cpp
    src/audio/asrc.cpp
    src/audio/asrc.hpp
    src/audio/audio.cpp
    src/audio/audioBlock.cpp
    src/audio/audiogen.cpp
    src/audio/audioPipeFactory.cpp
    src/audio/audioPipeline.cpp
    src/audio/converter.cpp
    src/audio/delayCalculator.cpp
    src/audio/envelopeDetector.cpp
    src/audio/filter.cpp
    src/audio/gain.cpp
    src/audio/resampler.cpp
    src/audio/sampleDelay.cpp
    src/audio/sigGen.cpp
    src/audio/wavReader.cpp
    src/audio/wavWriter.cpp
    src/audio/orah/orahAudioSync.cpp
    src/audio/orah/orahProcessor.cpp
    src/backend/common/glAllocator.cpp
    src/backend/common/core1/voronoi.cpp
    src/common/rwmutex.cpp
    src/common/semaphore.cpp
    src/common/thread.cpp
    src/core/bufferedReader.cpp
    src/core/controller.cpp
    src/core/controllerInputFrames.cpp
    src/core/defs/ambDecoderDef.cpp
    src/core/defs/audioPipeDef.cpp
    src/core/defs/cameraDef.cpp
    src/core/defs/controlPointListDef.cpp
    src/core/defs/controlPointListUpdater.cpp
    src/core/defs/curves.cpp
    src/core/defs/delayedAction.cpp
    src/core/defs/depthDef.cpp
    src/core/defs/geometryDef.cpp
    src/core/defs/readerInputDef.cpp
    src/core/defs/inputDef.cpp
    src/core/defs/inputDefinitionUpdater.cpp
    src/core/defs/mergerMaskDef.cpp
    src/core/defs/mergerMaskUpdater.cpp
    src/core/defs/overlayInputDef.cpp
    src/core/defs/overlayInputDefinitionUpdater.cpp
    src/core/defs/panoDef.cpp
    src/core/defs/panoramaDefinitionUpdater.cpp
    src/core/defs/ptoParse.cpp
    src/core/defs/rigCameraDef.cpp
    src/core/defs/rigDef.cpp
    src/core/defs/stereoRigDef.cpp
    src/core/emor.cpp
    src/core/geoProps.cpp
    src/core/geoTransform.cpp
    ${VERSION_HEADER}
    src/core/panoPipeline.cpp
    src/core/panoStitcherBase.cpp
    src/core/photoTransform.cpp
    src/core/postprocessor.cpp
    src/core/projections.cpp
    src/core/pyramid.cpp
    src/core/radial.cpp
    src/core/readerController.cpp
    src/core/rect.cpp
    src/core/rect.cpp
    src/core/stereoPipeline.cpp
    src/core/stitchOutput/algoOutput.cpp
    src/core/stitchOutput/asyncOutput.cpp
    src/core/stitchOutput/blockingOutput.cpp
    src/core/stitchOutput/frameBuffer.cpp
    src/core/stitchOutput/processorStitchOutput.cpp
    src/core/stitchOutput/stereoOutput.cpp
    src/core/stitchOutput/stitchOutput.cpp
    src/core/surfacePyramid.cpp
    src/core/transformGeoParams.cpp
    src/core/videoPipeline.cpp
    src/core1/arrayImageMerger.cpp
    src/core1/bounds.cpp
    src/core1/checkerboardImageMerger.cpp
    src/core1/diffImageMerger.cpp
    src/core1/exposureDiffImageMerger.cpp
    src/core1/gradientImageMerger.cpp
    src/core1/imageMapping.cpp
    src/core1/imageMerger.cpp
    src/core1/imageMergerFactory.cpp
    src/core1/inputsMap.cpp
    src/core1/inputsMapCubemap.cpp
    src/core1/laplacianImageMerger.cpp
    src/core1/maskMerger.cpp
    src/core1/noblendImageMerger.cpp
    src/core1/panoStitcher.cpp
    src/core1/stackImageMerger.cpp
    src/core1/voronoiMaskMerger.cpp
    src/coredepth/depthController.cpp
    src/coredepth/depthPipeline.cpp
    src/coredepth/depthStitcher.cpp
    src/coredepth/panoMerger.cpp
    src/coredepth/sphereSweep.cpp
    src/coredepth/sphereSweepMerger.cpp
    src/exposure/exposureStabilize.cpp
    src/exposure/metadataProcessor.cpp
    src/exposure/photometricCalibration.cpp
    src/exposure/pointSampler.cpp
    src/exposure/sampledStabilization.cpp
    src/gpu/surface2.cpp
    src/gpu/2dBuffer.cpp
    src/gpu/image/downsampler.cpp
    src/gpu/render/numberDrafter.cpp
    src/gpu/stream.cpp
    src/image/unpackShared.cpp
    src/input/checkerBoardReader.cpp
    src/input/colorReader.cpp
    src/input/exprReader.cpp
    src/input/input.cpp
    src/input/inputFactory.cpp
    src/input/maskedReader.cpp
    src/input/movingCheckerReader.cpp
    src/input/proceduralParser.cpp
    src/input/profilingReader.cpp
    src/input/statefulReader.cpp
    src/IO/sink.cpp
    src/motion/affineMotion.cpp
    src/motion/opticalFlow.cpp
    src/motion/rotationalMotion.cpp
    src/output/anaglyph.cpp
    src/output/compositeOutput.cpp
    src/output/output.cpp
    src/output/outputEventManager.cpp
    src/output/selectionOutput.cpp
    src/parallax/imageFlow.cpp
    src/parallax/imageFlowFactory.cpp
    src/parallax/imageWarper.cpp
    src/parallax/imageWarperFactory.cpp
    src/parallax/mergerPair.cpp
    src/parallax/noFlow.cpp
    src/parallax/noWarper.cpp
    src/parallax/sgm.cpp
    src/parse/helpers.cpp
    src/parse/json.cpp
    src/parse/jsonDriver.cpp
    src/parse/ptv.cpp
    src/parse/ubjson.cpp
    src/processors/exprProcessor.cpp
    src/processors/gridProcessor.cpp
    src/processors/maskProcessor.cpp
    src/processors/photoCorrProcessor.cpp
    src/processors/preprocessor.cpp
    src/processors/tintProcessor.cpp
    src/score/exposureScoring.cpp
    src/score/exposureScoringProcessor.cpp
    src/score/scoring.cpp
    src/score/scoringProcessor.cpp
    src/stabilization/iirFilter.cpp
    src/stabilization/imuStabilization.cpp
    src/stabilization/rotationStabilization.cpp
    src/stabilization/ukfQuaternion.cpp
    src/synchro/motionSync.cpp
    src/synchro/motionSyncFarneback.cpp
    src/synchro/sequencePeaks.cpp
    src/synchro/soundOffset.cpp
    src/undistort/undistortController.cpp
    src/undistort/undistortPipeline.cpp
    src/undistort/inputPipeline.cpp
    src/undistort/overrideDef.cpp
    src/util/base64.cpp
    src/util/compressionUtils.cpp
    src/util/drawingUtils.cpp
    src/util/dirutils.cpp
    src/util/expression.cpp
    src/util/fftsg_h.c
    src/util/geometryProcessingUtils.cpp
    src/util/imageProcessingUtils.cpp
    src/util/lmfit/lmmin.c
    src/util/logging.cpp
    src/util/matrix.cpp
    src/util/pixelFormat.cpp
    src/util/plugin.cpp
    src/util/pngutil.cpp
    src/util/pnm.cpp
    src/util/polylineEncodingUtils.cpp
    src/util/profile.cpp
    src/util/status.cpp
    src/util/strutils.cpp
    )

if(NOT ANDROID)
  set(CORE_LIB_SOURCES
      ${CORE_LIB_SOURCES}
      src/overlay/overlayer.cpp
  )
  if(NOT DISABLE_EXR)
    set(CORE_LIB_SOURCES
        ${CORE_LIB_SOURCES}
        src/util/exrutil.cpp
        )
  endif()
endif()

set(CORE_LIB_HEADERS
    src/audio/asrc.hpp
    src/audio/audiogen.hpp
    src/audio/audioPipeFactory.hpp
    src/audio/audioPipeline.hpp
    src/audio/audioPreProcessor.hpp
    src/audio/converter.hpp
    src/audio/delayCalculator.hpp
    src/audio/envelopeDetector.hpp
    src/audio/filter.hpp
    src/audio/gain.hpp
    src/audio/resampler.hpp
    src/audio/sampleDelay.hpp
    src/audio/sigGen.hpp
    src/audio/summer.hpp
    src/audio/orah/orahAudioSync.hpp
    src/audio/orah/orahProcessor.hpp
    src/bilateral/bilateral.hpp
    src/backend/cpp/core/transformStack.hpp
    src/backend/cpp/core/transformTypes.hpp
    src/backend/common/glAllocator.hpp
    src/backend/common/allocStats.hpp
    src/backend/common/core1/voronoiKernel.hpp
    src/backend/common/core/transformPhotoParam.hpp
    src/backend/common/core/types.hpp
    src/backend/common/coredepth/sphereSweepParams.h
    src/backend/common/imageOps.hpp
    src/backend/common/vectorOps.hpp
    src/common/angles.hpp
    src/common/container.hpp
    src/common/graph.hpp
    src/common/queue.hpp
    src/common/rwmutex.hpp
    src/common/thread.hpp
    src/core/bufferedReader.hpp
    src/core/controller.hpp
    src/core/controllerInputFrames.hpp
    src/core/coordinates.hpp
    src/core/defs/panoInputDefsPimpl.hpp
    src/core/emor_coefs.hpp
    src/core/emor_inv_coefs.hpp
    src/core/geoProps.hpp
    src/core/geoTransform.hpp
    src/core/inputControllerImpl.hpp
    src/core/panoDimensions.hpp
    src/core/panoDimensionsDef.hpp
    src/core/panoPipeline.hpp
    src/core/panoStitcherBase.hpp
    src/core/photoTransform.hpp
    src/core/pyramid.hpp
    src/core/quarticSolver.hpp
    src/core/radial.hpp
    src/core/readerController.hpp
    src/core/rect.hpp
    src/core/stereoPipeline.hpp
    src/core/stitchOutput/asyncOutput.hpp
    src/core/stitchOutput/blockingOutput.hpp
    src/core/stitchOutput/frameBuffer.hpp
    src/core/stitchOutput/stereoOutput.hpp
    src/core/stitchOutput/stitchOutput.hpp
    src/core/surfacePyramid.hpp
    src/core/transformGeoParams.hpp
    src/core/videoPipeline.hpp
    src/core1/arrayImageMerger.hpp
    src/core1/bounds.hpp
    src/core1/checkerboardImageMerger.hpp
    src/core1/diffImageMerger.hpp
    src/core1/exposureDiffImageMerger.hpp
    src/core1/gradientImageMerger.hpp
    src/core1/imageMapping.hpp
    src/core1/imageMerger.hpp
    src/core1/inputsMap.hpp
    src/core1/inputsMapCubemap.hpp
    src/core1/laplacianImageMerger.hpp
    src/core1/maskMerger.hpp
    src/core1/noblendImageMerger.hpp
    src/core1/panoRemapper.hpp
    src/core1/panoStitcher.hpp
    src/core1/stackImageMerger.hpp
    src/core1/voronoiMaskMerger.hpp
    src/coredepth/depthControllerImpl.hpp
    src/coredepth/depthPipeline.hpp
    src/coredepth/depthStitcher.hpp
    src/coredepth/panoMerger.hpp
    src/coredepth/sphereSweep.hpp
    src/coredepth/sphereSweepMerger.hpp
    src/cuda/error.hpp
    src/cuda/memory.hpp
    src/cuda/util.hpp
    src/exposure/exposureStabilize.hpp
    src/exposure/metadataProcessor.hpp
    src/exposure/photometricCalibration.hpp
    src/exposure/pointSampler.hpp
    src/exposure/sampledStabilization.hpp
    src/gpu/buffer.hpp
    src/gpu/core1/boundsKernel.hpp
    src/gpu/core1/eRectTransform.hpp
    src/gpu/core1/gpuPanoRemapper.hpp
    src/gpu/core1/mergerKernel.hpp
    src/gpu/core1/strip.hpp
    src/gpu/core1/transform.hpp
    src/gpu/core1/voronoi.hpp
    src/gpu/coredepth/sweep.hpp
    src/gpu/event.hpp
    src/gpu/hostBuffer.hpp
    src/gpu/image/blur.hpp
    src/gpu/image/downsampler.hpp
    src/gpu/image/imageOps.hpp
    src/gpu/image/imgExtract.hpp
    src/gpu/image/imgInsert.hpp
    src/gpu/image/reduce.hpp
    src/gpu/image/sampling.hpp
    src/gpu/input/checkerBoard.hpp
    src/gpu/input/maskInput.hpp
    src/gpu/memcpy.hpp
    src/gpu/processors/grid.hpp
    src/gpu/processors/maskoverlay.hpp
    src/gpu/processors/photoCorr.hpp
    src/gpu/processors/tint.hpp
    src/gpu/render/geometry.hpp
    src/gpu/render/numberDrafter.hpp
    src/gpu/render/render.hpp
    src/gpu/score/scoring.hpp
    src/gpu/stream.hpp
    src/gpu/typeTraits.hpp
    src/gpu/uniqueBuffer.hpp
    src/gpu/util.hpp
    src/gpu/vectorTypes.hpp
    src/image/colorArray.hpp
    src/image/fill.hpp
    src/image/histogram.hpp
    src/image/histogramView.hpp
    src/image/kernels/sharedUtils.hpp
    src/image/transpose.hpp
    src/image/unpack.hpp
    include/libvideostitch/algorithm.hpp
    include/libvideostitch/allocator.hpp
    include/libvideostitch/ambisonic.hpp
    include/libvideostitch/ambDecoderDef.hpp
    include/libvideostitch/audio.hpp
    include/libvideostitch/audioBlock.hpp
    include/libvideostitch/audioObject.hpp
    include/libvideostitch/audioPipeDef.hpp
    include/libvideostitch/audioWav.hpp
    include/libvideostitch/cameraDef.hpp
    include/libvideostitch/config.hpp
    include/libvideostitch/context.hpp
    include/libvideostitch/controller.hpp
    include/libvideostitch/controlPointListDef.hpp
    include/libvideostitch/controlPointListUpdater.hpp
    include/libvideostitch/curves.hpp
    include/libvideostitch/deferredUpdater.hpp
    include/libvideostitch/delayedAction.hpp
    include/libvideostitch/depthController.hpp
    include/libvideostitch/depthDef.hpp
    include/libvideostitch/dirutils.hpp
    include/libvideostitch/emor.hpp
    include/libvideostitch/geometryDef.hpp
    include/libvideostitch/gpu_device.hpp
    include/libvideostitch/imageFlowFactory.hpp
    include/libvideostitch/imageMergerFactory.hpp
    include/libvideostitch/imageMergerFactory.hpp
    include/libvideostitch/imageProcessingUtils.hpp
    include/libvideostitch/imageWarperFactory.hpp
    include/libvideostitch/imuData.hpp
    include/libvideostitch/input.hpp
    include/libvideostitch/sink.hpp
    include/libvideostitch/span.hpp
    include/libvideostitch/circularBuffer.hpp
    include/libvideostitch/readerInputDef.hpp
    include/libvideostitch/inputController.hpp
    include/libvideostitch/inputDef.hpp
    include/libvideostitch/inputDefinitionUpdater.hpp
    include/libvideostitch/inputFactory.hpp
    include/libvideostitch/logging.hpp
    include/libvideostitch/matrix.hpp
    include/libvideostitch/mergerMaskDef.hpp
    include/libvideostitch/mergerMaskUpdater.hpp
    include/libvideostitch/orah/exposureData.hpp
    include/libvideostitch/orah/imuStabilization.hpp
    include/libvideostitch/object.hpp
    include/libvideostitch/opengl.hpp
    include/libvideostitch/output.hpp
    include/libvideostitch/outputEventManager.hpp
    include/libvideostitch/overlayInputDef.hpp
    include/libvideostitch/overlayInputDefinitionUpdater.hpp
    include/libvideostitch/overrideDef.hpp
    include/libvideostitch/panoDef.hpp
    include/libvideostitch/panoramaDefinitionUpdater.hpp
    include/libvideostitch/parse.hpp
    include/libvideostitch/plugin.hpp
    include/libvideostitch/postprocessor.hpp
    include/libvideostitch/preprocessor.hpp
    include/libvideostitch/processorStitchOutput.hpp
    include/libvideostitch/profile.hpp
    include/libvideostitch/projections.hpp
    include/libvideostitch/ptv.hpp
    include/libvideostitch/quaternion.hpp
    include/libvideostitch/rigCameraDef.hpp
    include/libvideostitch/rigDef.hpp
    include/libvideostitch/status.hpp
    include/libvideostitch/stereoRigDef.hpp
    include/libvideostitch/stitchOutput.hpp
    include/libvideostitch/undistortController.hpp
    include/libvideostitch/utils/semaphore.hpp
    src/input/checkerBoardReader.hpp
    src/input/colorReader.hpp
    src/input/exprReader.hpp
    src/input/inputFrame.hpp
    src/input/maskedReader.hpp
    src/input/movingCheckerReader.hpp
    src/input/proceduralParser.hpp
    src/input/profilingReader.hpp
    src/input/statefulReader.hpp
    src/mask/dijkstraShortestPath.hpp
    src/mask/mergerMask.hpp
    src/mask/mergerMaskAlgorithm.hpp
    src/mask/mergerMaskConfig.hpp
    src/mask/mergerMaskConstant.hpp
    src/mask/mergerMaskProgress.hpp
    src/mask/seamFinder.hpp
    src/motion/affineMotion.hpp
    src/motion/gme.hpp
    src/motion/opticalFlow.hpp
    src/motion/rotationalMotion.hpp
    src/output/anaglyph.hpp
    src/output/compositeOutput.hpp
    src/output/discardOutput.hpp
    src/output/profilingOutput.hpp
    src/parallax/imageFlow.hpp
    src/parallax/imageWarper.hpp
    src/parallax/mergerPair.hpp
    src/parallax/noFlow.hpp
    src/parallax/noWarper.hpp
    src/parallax/sgm.hpp
    src/parse/json.hpp
    src/parse/jsonDriver.hpp
    src/processors/exprProcessor.hpp
    src/processors/maskProcessor.hpp
    src/processors/photoCorrProcessor.hpp
    src/processors/tintProcessor.hpp
    src/render/fillRenderer.hpp
    src/score/exposureScoring.hpp
    src/score/exposureScoringProcessor.hpp
    src/score/scoring.hpp
    src/score/scoringProcessor.hpp
    src/stabilization/iirFilter.hpp
    src/stabilization/ukfQuaternion.hpp
    src/stabilization/rotationStabilization.hpp
    src/synchro/flashSync.hpp
    src/synchro/motionSync.hpp
    src/synchro/motionSyncFarneback.hpp
    src/synchro/sequencePeaks.hpp
    src/synchro/soundOffset.hpp
    src/undistort/inputPipeline.hpp
    src/undistort/undistortControllerImpl.hpp
    src/undistort/undistortPipeline.hpp
    src/util/base64.hpp
    src/util/compressionUtils.hpp
    src/util/drawingUtils.hpp
    src/util/debugUtils.hpp
    src/util/debugUtils.hpp
    src/util/expression.hpp
    src/util/geometryProcessingUtils.hpp
    src/util/lmfit/lmmin.hpp
    src/util/plugin.hpp
    src/util/exrutil.hpp
    src/util/pngutil.hpp
    src/util/pnm.hpp
    src/util/polylineEncodingUtils.hpp
    src/util/ransac.hpp
    src/util/registeredAlgo.hpp
    src/util/strutils.hpp
    src/overlay/oglShader.hpp
    src/overlay/overlayer.hpp
    )

# https://cmake.org/cmake/help/v3.6/command/add_library.html
# "Some native build systems may not like targets that have only object files, so consider adding at least one real source file to any target that references $<TARGET_OBJECTS:objlib>."
# --> Adding a single source file here instead of to the object lib, so it compiles in Xcode.
set(NON_OBJECT_SOURCE
    src/util/algorithm.cpp)

# ----------------------------------------------------------------------------
# GPU backend specific CMake configuration
# ----------------------------------------------------------------------------
if(GPU_BACKEND_OPENCL)
  include(libvideostitch_OpenCL.cmake)
endif()
if(GPU_BACKEND_CUDA)
  include(libvideostitch_CUDA.cmake)
endif()

# ----------------------------------------------------------------------------
# r8b audio resampling lib
# ----------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
  r8b
  GIT_REPOSITORY https://github.com/avaneev/r8brain-free-src.git
  GIT_TAG        7f1980ecf26be3a2424aa46c68577f12f94a00f0
)

FetchContent_GetProperties(r8b)
if(NOT r8b_POPULATED)
  FetchContent_Populate(r8b)
  file(COPY ${CMAKE_SOURCE_DIR}/cmake/r8b/CMakeLists.txt
       DESTINATION ${r8b_SOURCE_DIR})
  add_subdirectory(${r8b_SOURCE_DIR} ${r8b_BINARY_DIR})
  get_filename_component(DEPS_DOWNLOAD_FOLDER ${r8b_SOURCE_DIR} DIRECTORY)
  set(VS_LIB_SYSTEM_INCLUDES ${VS_LIB_SYSTEM_INCLUDES} ${DEPS_DOWNLOAD_FOLDER})
endif()

# ----------------------------------------------------------------------------
# Calibration && AutoCrop && CameraAmbisonics
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

add_subdirectory(src/calibration)
add_subdirectory(src/autocrop)
add_subdirectory(src/audio/orah/orah4i2b)
add_subdirectory(src/epipolar)

# ----------------------------------------------------------------------------
message(STATUS "--- VideoStitch-lib dependencies ---")

# ----------------------------------------------------------------------------
# EXR support
# ----------------------------------------------------------------------------
if(DISABLE_EXR)
  message(STATUS "OpenEXR disabled")
else()
  find_package(OpenEXR REQUIRED)
  message(STATUS "OpenEXR libs: ${OpenEXR_LIBRARIES}")
  message(STATUS "OpenEXR include: ${OpenEXR_INCLUDE_DIRS}")
endif()

# ----------------------------------------------------------------------------
message(STATUS "PNG: ${PNG_LIBRARY}")
message(STATUS "R8B: ${R8B_LIBRARY}")
message(STATUS "OpenCV libs: ${OpenCV_LIBRARIES}")
message(STATUS "CERES LIBS: ${CERES_LIBS}")
message(STATUS "")


# ----------------------------------------------------------------------------
# libvideostitch target
# ----------------------------------------------------------------------------
if(APPLE)
  vs_lib_link_libraries("COMMON" ${CORE_FOUNDATION} ${IO_KIT})
elseif (NOT ANDROID)
  vs_lib_link_libraries("PUBLIC_COMMON" Threads::Threads)
endif(APPLE)
if(TEGRA_DEMO OR WINDOWS)
    vs_lib_link_libraries("PUBLIC_COMMON" ${GLOG})
endif()
vs_lib_link_libraries("COMMON" ${OpenCV_LIBRARIES})
vs_lib_link_libraries("COMMON" ${PNG_LIBRARY} ${CERES_LIBS} ${HID} ${R8B_LIBRARY})
vs_lib_link_libraries("COMMON" o2b)
vs_lib_link_libraries("COMMON" ${VS_DISCOVERY})
if(NOT DISABLE_EXR)
  vs_lib_link_libraries("PUBLIC_COMMON" ${OpenEXR_LIBRARIES})
endif()
if(LINUX AND NOT TEGRA_DEMO)
  find_library(DRM NAMES "drm")
  if(NOT DRM)
    message(FATAL_ERROR "libdrm not found")
  endif()
  vs_lib_link_libraries("PUBLIC_COMMON" ${DRM})
endif()

function(configure_lib_main_target VS_LIB_TARGET BACKEND_NAME LIB_TYPE)
  add_library(${VS_LIB_TARGET} ${LIB_TYPE} ${NON_OBJECT_SOURCE} ${BACKEND_OBJECTS_${BACKEND_NAME}} $<TARGET_OBJECTS:${VS_LIB_OBJECTS_${BACKEND_NAME}}> $<TARGET_OBJECTS:calibration_${BACKEND_NAME}> $<TARGET_OBJECTS:autocrop_${BACKEND_NAME}> $<TARGET_OBJECTS:epipolar_${BACKEND_NAME}>)
  add_cppcheck(${VS_LIB_TARGET} VS)

  include_discovery_vs_headers(${VS_LIB_TARGET})
  target_include_directories(${VS_LIB_TARGET} PRIVATE src)
  target_compile_definitions(${VS_LIB_TARGET} PRIVATE VS_LIB_COMPILATION _USE_MATH_DEFINES)
  target_compile_definitions(${VS_LIB_TARGET} PRIVATE VS_${BACKEND_NAME})
  include_lib_vs_headers(${VS_LIB_TARGET})
  set_property(TARGET ${VS_LIB_TARGET} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${VS_LIB_TARGET} PROPERTY FOLDER "lib")

  if(("${LIB_TYPE}" STREQUAL "SHARED") AND (LINUX OR APPLE))
    set_property(TARGET ${VS_LIB_TARGET} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endif()
  if(WINDOWS)
    set_property(TARGET ${VS_LIB_TARGET} APPEND_STRING PROPERTY LINK_FLAGS "/NODEFAULTLIB:libcmt /NODEFAULTLIB:libcmtd")
  endif(WINDOWS)

  set_iwyu(${VS_LIB_TARGET})
  vs_lib_target_link_libraries(${VS_LIB_TARGET})
endfunction(configure_lib_main_target)

function(configure_lib_targets BACKEND_NAME)
  set_property(TARGET ${VS_LIB_OBJECTS_${BACKEND_NAME}} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${VS_LIB_OBJECTS_${BACKEND_NAME}} PROPERTY FOLDER "lib")

  include_discovery_vs_headers(${VS_LIB_OBJECTS_${BACKEND_NAME}})
  include_lib_vs_headers(${VS_LIB_OBJECTS_${BACKEND_NAME}})
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE src)
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} SYSTEM PRIVATE ${VS_LIB_SYSTEM_INCLUDES})
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} SYSTEM PRIVATE ${CMAKE_EXTERNAL_DEPS}/include)
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS})
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE ${JSON_PARSER_ROOT_DIR})
  # TODO: only needed for ukfQuaternion
  target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE ${EIGEN3_INCLUDE_DIRS})
  target_compile_definitions(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE VS_${BACKEND_NAME})

  if(LINUX OR APPLE)
    set_property(TARGET ${VS_LIB_OBJECTS_${BACKEND_NAME}} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endif(LINUX OR APPLE)

  if(WINDOWS)
    target_compile_definitions(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE VS_LIB_COMPILATION _USE_MATH_DEFINES)
    target_compile_options(${VS_LIB_OBJECTS_${BACKEND_NAME}} PRIVATE "/bigobj")
  endif(WINDOWS)

  if(NOT DISABLE_EXR)
    target_include_directories(${VS_LIB_OBJECTS_${BACKEND_NAME}} SYSTEM PRIVATE ${OpenEXR_INCLUDE_DIRS})
  endif()

  configure_lib_main_target(${VS_LIB_${BACKEND_NAME}} ${BACKEND_NAME} "SHARED")
endfunction()

if(GPU_BACKEND_CUDA)
  configure_lib_targets("CUDA")
endif()
if(GPU_BACKEND_OPENCL)
  configure_lib_targets("OPENCL")
endif()


# ----------------------------------------------------------------------------
# Fake libvideostitch
# The target VS_LIB_FAKE will be built only when we have both backends on Windows or on Apple.
# It will ensure that we have a neutral libvideostitch library.
# To be sure, the generated dll on Windows will be replaced by an empty one in generateFakeLibvideostitch,
# while VS_LIB will be loaded using the delay load function
# ----------------------------------------------------------------------------
if(USE_DELAY_LOAD)
  # Use any backend, we don't care, we just want the .lib with the common exported symbols
  configure_lib_main_target(${VS_LIB_FAKE} ${GPU_BACKEND_DEFAULT} "SHARED")

  if(MSVC)
    function(generate_lib_symbols BACKEND_NAME LIB_DIR)
      add_custom_command(TARGET ${VS_LIB_${BACKEND_NAME}}
                         POST_BUILD
                         COMMAND dumpbin /EXPORTS /OUT:lib_symbols_${BACKEND_NAME}.txt ${LIB_DIR}/${VS_LIB_${BACKEND_NAME}}.lib
                         COMMAND cat lib_symbols_${BACKEND_NAME}.txt | grep ? > lib_symbols_only_${BACKEND_NAME}.txt)
    endfunction()
    # Check that VS_LIB_CUDA and VS_LIB_OPENCL have the same symbols
    if(CMAKE_BUILD_TYPE)
      generate_lib_symbols("CUDA" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      generate_lib_symbols("OPENCL" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
      generate_lib_symbols("CUDA" ${VS_OUT_DIR}/${CMAKE_CFG_INTDIR})
      generate_lib_symbols("OPENCL" ${VS_OUT_DIR}/${CMAKE_CFG_INTDIR})
    endif(CMAKE_BUILD_TYPE)

    add_custom_target(compareLibsSymbols ALL
                      COMMAND ${CMAKE_COMMAND} -E compare_files lib_symbols_only_CUDA.txt lib_symbols_only_OPENCL.txt
                      DEPENDS ${VS_LIB_CUDA} ${VS_LIB_OPENCL})
  endif()
endif()

# ----------------------------------------------------------------------------
# static lib
# ----------------------------------------------------------------------------
if(BUILD_STATIC_LIB)
  configure_lib_main_target(${VS_LIB_STATIC} ${GPU_BACKEND_DEFAULT} "STATIC")
  if(MSVC)
    add_dependencies(${VS_LIB_STATIC} ${VS_LIB_${GPU_BACKEND_DEFAULT}})
  endif(MSVC)
endif(BUILD_STATIC_LIB)

# ----------------------------------------------------------------------------
# videostitch-cmd target
# ----------------------------------------------------------------------------
add_subdirectory(samples/sample-02-fullCmd)

# ----------------------------------------------------------------------------
# autocrop-cmd target
# ----------------------------------------------------------------------------
add_subdirectory(samples/autocrop)

# ----------------------------------------------------------------------------
# calibrationimport target
# ----------------------------------------------------------------------------
add_subdirectory(samples/calibrationimport)

# ----------------------------------------------------------------------------
# ptvb2ptv target
# ----------------------------------------------------------------------------
add_subdirectory(samples/ptvb2ptv)

# ----------------------------------------------------------------------------
# ambisonic encoder tool target
# ----------------------------------------------------------------------------
add_subdirectory(samples/ambisonic)

# ----------------------------------------------------------------------------
# undistortion tool target
# ----------------------------------------------------------------------------
add_subdirectory(samples/undistort)

# ----------------------------------------------------------------------------
# depth estimation tool target
# ----------------------------------------------------------------------------
add_subdirectory(samples/depth)


# ----------------------------------------------------------------------------
# libvideostitch unit tests
# ----------------------------------------------------------------------------
if(LIB_TESTS)
  add_subdirectory(src/test)
endif()

# Add filter to MSVS project
assign_source_group(${CORE_LIB_SOURCES} ${CORE_LIB_HEADERS})
assign_source_group(${CUDA_SOURCES})
assign_source_group(${CUDA_BACKEND_SOURCES} ${CUDA_BACKEND_HEADERS})

# ----------------------------------------------------------------------------
# Create documentation
# ----------------------------------------------------------------------------
find_host_package(Doxygen)
add_custom_target(doc
  ${DOXYGEN_EXECUTABLE} "doc/Doxyfile-library"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  COMMENT "Generating API documentation with Doxygen" VERBATIM)
set_property(TARGET doc PROPERTY FOLDER "lib")

# ----------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------
if(GENERATE_BINDINGS)
  add_subdirectory(bindings)
endif()

