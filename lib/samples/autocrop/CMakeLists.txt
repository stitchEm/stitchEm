# ----------------------------------------------------------------------------
# Safeguards against invalid configurations
# ----------------------------------------------------------------------------
if(NOT VIDEOSTITCH_CMAKE)
  message(FATAL_ERROR "Please configure CMake from the root folder!")
endif(NOT VIDEOSTITCH_CMAKE)

find_package(OpenCV)

# ----------------------------------------------------------------------------
# Auto-crop
# To enable/disable the Autocrop : -DAUTOCROP_CMD=ON or -DAUTOCROP_CMD=OFF (by default)
# ----------------------------------------------------------------------------
option(AUTOCROP_CMD "Build VideoStitch auto crop" ON)

# ----------------------------------------------------------------------------
# autocrop-cmd target
# ----------------------------------------------------------------------------
if(AUTOCROP_CMD)
  set(CMD_NAME "autocrop-cmd")
  add_executable(${CMD_NAME} autocrop-cmd.cpp)
  target_compile_definitions(${CMD_NAME} PRIVATE VS_LIB_COMPILATION _USE_MATH_DEFINES)
  set_property(TARGET ${CMD_NAME} PROPERTY FOLDER "lib/samples")

  set_property(TARGET ${CMD_NAME} PROPERTY CXX_STANDARD 14)
  include_lib_vs_headers(${CMD_NAME})
  link_target_to_libvideostitch(${CMD_NAME})
  target_include_directories(${CMD_NAME} SYSTEM PRIVATE "${CMAKE_EXTERNAL_DEPS}/include")
  target_include_directories(${CMD_NAME} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
endif(AUTOCROP_CMD)

