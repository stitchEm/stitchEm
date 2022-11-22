# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)


# ----------------------------------------------------------------------------
# Ambisonic encoder tool
# To enable/disable the Ambisonic encoder tool :
# -DAMBISONIC_TOOL=ON or -DAMBISONIC_TOOL=OFF (by default)
# ----------------------------------------------------------------------------
option(AMBISONIC_ENCODER_TOOL "Build VideoStitch ambisonic encoder tools" ON)

# ----------------------------------------------------------------------------
# aec target
# ----------------------------------------------------------------------------
if(AMBISONIC_ENCODER_TOOL)
  add_executable(aec ambisonicEncoderTool.cpp)
  include_lib_vs_headers(aec)
  link_target_to_libvideostitch(aec)
endif()

