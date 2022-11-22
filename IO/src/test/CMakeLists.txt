# ----------------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------------

# Set test binary output dir for the generic no-config case (e.g. make, ninja)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_TEST_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_TEST_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_TEST_DIR})

# Set test binary output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
endforeach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})

# ----------------------------------------------------------------------------


get_filename_component(TEST_IO_SRC_PATH . ABSOLUTE)

function(add_vs_io_test test_name test_file)
  add_executable (${test_name} ${test_file})
  add_definitions(-DVS_PLUGIN_DIR_NAME="${VS_VAHANA_PLUGIN_DIR_NAME}")
  if(WINDOWS)
    target_link_libraries (${test_name} ${VS_LIB_STATIC})
    target_compile_definitions(${test_name} PRIVATE VS_LIB_COMPILATION NOMINMAX _USE_MATH_DEFINES)
  else(WINDOWS)
    target_link_libraries (${test_name} ${VS_LIB_DEFAULT})
  endif(WINDOWS)

  target_include_directories(${test_name} PRIVATE ../include)
  target_include_directories(${test_name} PRIVATE ${TESTING_INCLUDE})
  include_lib_vs_headers(${test_name})
  foreach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
    target_include_directories(${test_name} SYSTEM PRIVATE ${vs_lib_sys_include})
  endforeach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endfunction(add_vs_io_test test_name test_file)

if(STAGING)
    # XXX: ADD HERE NEW TEST
else(STAGING)
  if(TEGRA_DEMO)
    if(ANDROID)
      #add_vs_io_test(MP4Test mp4Test.cpp)
    endif(ANDROID)
      #add_vs_io_test(BmpTest bmpTest.cpp)
      #add_vs_io_test(RawTest rawTest.cpp)
  else(TEGRA_DEMO)
    #add_vs_io_test(AvTest avTest.cpp)
    #add_vs_io_test(JpgTest jpgTest.cpp)
    #add_vs_io_test(PamTest pamTest.cpp)
    #add_vs_io_test(PngTest pngTest.cpp)
    #add_vs_io_test(RawTest rawTest.cpp)
    #add_vs_io_test(TiffTest tiffTest.cpp)
    # add_vs_io_test(MetaDataParsingTest orahMetadataParsingTest.cpp)
    if(LINUX OR WINDOWS)
      #add_vs_io_test(RtmpTest rtmpTest.cpp)
      #add_vs_io_test(PortAudioTest portaudioTest.cpp)
    endif(LINUX OR WINDOWS)
  endif(TEGRA_DEMO)
endif(STAGING)
