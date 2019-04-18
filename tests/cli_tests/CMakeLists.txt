option(ALL_CLI_TESTS "Launch all command line tests" OFF)

# TODO: run simple command line tests (that don't allocate huge resources) in parallel
function(add_cli_tests TESTS_TO_ADD)
  foreach(test ${TESTS_TO_ADD})
    add_test(
      NAME "${test}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/cli_tests"
      COMMAND "behave" -t=-slow -D "BIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" -D GPU=${GPU_BACKEND_DEFAULT} "features/${test}")
    set_property(TEST ${test} PROPERTY RUN_SERIAL TRUE)
    set_property(TEST ${test} PROPERTY FOLDER "cli tests")
  endforeach()
endfunction(add_cli_tests TESTS_TO_ADD)
function(add_slow_cli_tests TESTS_TO_ADD)
  foreach(test ${TESTS_TO_ADD})
    add_test(
      NAME "${test}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/cli_tests"
      COMMAND "behave" -D "BIN=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" -D GPU=${GPU_BACKEND_DEFAULT} "features/${test}")
    set_property(TEST ${test} PROPERTY RUN_SERIAL TRUE)
    set_property(TEST ${test} PROPERTY FOLDER "cli tests")
  endforeach()
endfunction(add_slow_cli_tests TESTS_TO_ADD)

# fast tests working with either GPU backend
set(TESTS
    audio_format.feature
    audio_synchro.feature
    autocrop.feature
    autocrop_cmd.feature
    calibration_apply_presets_without_calibrating.feature
    cmd.feature
    gpu_compatibility.feature
    gradient.feature
    mapping.feature
    mixed_format.feature
    synchro.feature
    )

if(NOT OCLGRIND)
  set(TESTS
      ${TESTS}
      calibration_deshuffle.feature
      calibration_procedurals.feature
      )
endif()


if(NOT WINDOWS AND NOT OCLGRIND)
  set(TESTS
      ${TESTS}
      memory.feature)
endif()

#enable x server for overlay test
include(EnableXServer)

# The GPU on the bbmac (NV GeForce 640) is obsolete
# BUGS-41, clFinish issue on bb windows_opencl
if((NOT (${GPU_BACKEND_DEFAULT} STREQUAL CUDA AND APPLE)) AND (NOT (${GPU_BACKEND_DEFAULT} STREQUAL OPENCL AND WINDOWS)))
  if(HAVE_DISPLAY AND NOT OCLGRIND)
    set(TESTS
        ${TESTS}
        overlay.feature)
  endif()
endif()

# JIRA: CL-74
if((NOT LINUX OR ${GPU_BACKEND_DEFAULT} STREQUAL CUDA) AND NOT OCLGRIND)
  # TB-182
  if(NOT APPLE OR ${GPU_BACKEND_DEFAULT} STREQUAL CUDA)
    set(TESTS
        ${TESTS}
        rtmp_client.feature
        )
  endif()
  set(TESTS
      ${TESTS}
      #TB-184
      audio_channel_map.feature
      bilinear.feature
      calibration_failing.feature
      calibration_presets_maker.feature
      calibration_preserve_translations.feature
      #FI-1071
      exposure.feature
      input_formats.feature
      mergers.feature
      pano_coverage.feature
      photometric.feature
      precomputed_coord_gradient_blending.feature
      rtmp_publisher.feature
      tiny_pano.feature
      undistort.feature
      )
  if(${GPU_BACKEND_DEFAULT} STREQUAL CUDA OR NOT WINDOWS)
    set(TESTS
        ${TESTS}
        calibration.feature
        calibration_auto_fov.feature
        )
  endif()
  set(TESTS
      ${TESTS}
      curves.feature
      exposure_algo.feature
      exposure_correction.feature
      seek_frame.feature
      )
endif()

# fast tests only working with CUDA
set(CUDA_TESTS
    arg.feature                  # OpenCL TODO: OpenCL TODO: uses laplacian blending
    calibration_orah_4i.feature  # OpenCL TODO: test failing because sphere scale is not used by _opencl builds
    depth_sweep.feature          # OpenCL TODO: algorithm not implemented
    flow_based_blending.feature  # OpenCL TODO: algorithm not implemented
    frame_offset.feature         # OpenCL TODO: uses laplacian blending
    image_format.feature         # OpenCL TODO: uses FullFrameFisheye (templatePanoRemapper)
    laplacian_blending.feature   # OpenCL TODO: uses laplacian blending
    secondary_gpu.feature        # OpenCL TODO: uses laplacian blending
    precomputed_coord_blending.feature   # OpenCL TODO:
    transform.feature            # OpenCL TODO: needs checkerboard + projections
    )
if(NOT CMAKE_BUILD_TYPE STREQUAL "LSan")
  set(CUDA_TESTS
      ${CUDA_TESTS}
      # VSA-7011
      error.feature              # OpenCL TODO: uses laplacian blending
      )
  set(TESTS
      ${TESTS}
      epipolar.feature
      )
endif()
if(NOT APPLE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "LSan")
    set(CUDA_TESTS
        h264_nvenc.feature           # NVENC is CUDA only
        hevc_nvenc.feature           # NVENC is CUDA only
        rtmp_nvenc.feature           # NVENC is CUDA only
        ${CUDA_TESTS}
        )
    if(LINUX)
      set(CUDA_TESTS
          youtube.feature            # NVENC is CUDA only
          ${CUDA_TESTS}
          )
    endif(LINUX)
  endif(NOT CMAKE_BUILD_TYPE STREQUAL "LSan")
endif(NOT APPLE)

# VSA-5751
if(NOT ((CMAKE_BUILD_TYPE STREQUAL "ASan") AND COMPILER_CLANG))
  set(TESTS
      ${TESTS}
      no_audio_track.feature
      )
endif()

# slow tests using linux tool to limit bandwith using trickle
# which does behave fine with lsan
if(LINUX AND (NOT CMAKE_BUILD_TYPE STREQUAL "LSan"))
  set(MORE_CUDA_TESTS
      ${MORE_CUDA_TESTS}
      rtmp_bandwidth.feature
      )
endif()

# Slow CUDA tests
set(CUDA_TESTS
    ${CUDA_TESTS}
    audio_output.feature
    audio_pipe.feature
    buffer.feature              # OpenCL TODO: uses laplacian blending
    formats_cuda.feature
    mask_blending.feature       # OpenCL TODO: mask algorithm not implemented
    mask_seam_blending.feature  # OpenCL TODO: mask algorithm not implemented
    procedural.feature           # OpenCL TODO: uses laplacian blending, procedurals
    unicode.feature              # ?
    verbose.feature              # OpenCL TODO: uses laplacian blending
    video_audio_filesize.feature # OpenCL TODO VSA-7069: crash on Intel GPU on macOS Sierra
    video_filesize.feature       # OpenCL TODO VSA-7069: crash on Intel GPU on macOS Sierra
    video_format.feature         # OpenCL TODO: uses laplacian blending
    )

set(OPENCL_TESTS           #TODO: remove this setting when OpenCL support laplacian blending
    formats_opencl.feature
    video_format_opencl.feature
    )

if(STAGING)
  # XXX: ADD HERE NEW TESTS
else(STAGING)

  if(ALL_CLI_TESTS)
    add_slow_cli_tests("${TESTS}")

    if(${GPU_BACKEND_DEFAULT} STREQUAL CUDA)
      add_slow_cli_tests("${CUDA_TESTS}")
    else()
      add_slow_cli_tests("${OPENCL_TESTS}")
    endif()

  else(ALL_CLI_TESTS)
    add_cli_tests("${TESTS}")

    if(${GPU_BACKEND_DEFAULT} STREQUAL CUDA)
      add_cli_tests("${CUDA_TESTS}")
    else()
      add_cli_tests("${OPENCL_TESTS}")
    endif()
  endif(ALL_CLI_TESTS)
endif(STAGING)
