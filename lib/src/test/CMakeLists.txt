# ----------------------------------------------------------------------------
# Output directories
# ----------------------------------------------------------------------------

# Set test binary output dir for the generic single-config case (e.g. make, ninja)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VS_TEST_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_TEST_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VS_TEST_DIR})

# Set test binary output dir for multi-config builds (e.g. MSVC, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# ----------------------------------------------------------------------------

# Assets folder
set(VS_ASSETS $ENV{VIDEOSTITCH_ASSETS})

#Folder to write tests data
set(VS_TEST_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/testdata)
file(MAKE_DIRECTORY ${VS_TEST_DATA_DIR})

if (WINDOWS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309 /wd4324")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
  STRING(REPLACE "INCREMENTAL" "INCREMENTAL:NO" CMAKE_EXE_LINKER_FLAGS_DEBUG ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
endif(WINDOWS)

get_filename_component(TEST_SRC_PATH . ABSOLUTE)

# All trailing arguments are libs to link
function(add_vs_unit_test test_name test_file)
  add_vs_test_arg(${test_name} ${test_file} "unit" "" ${ARGN})
endfunction()

# All trailing arguments are libs to link
function(add_vs_test_arg test_name test_file test_labels test_args)
  add_executable (${test_name} ${test_file} ${ADDITIONAL_SOURCES})

  if(WINDOWS)
    set_property(TARGET ${test_name} APPEND_STRING PROPERTY LINK_FLAGS "/IGNORE:4099 /IGNORE:4098")
    target_compile_definitions(${test_name} PRIVATE VS_LIB_COMPILATION _USE_MATH_DEFINES NOMINMAX)
  endif()

  if(OCLGRIND)
    target_compile_definitions(${test_name} PRIVATE "OCLGRIND")
  endif()

  set(LIBS ${ARGN})
  if(NO_VS_LIB_LINK)
      target_link_libraries(${test_name} PRIVATE ${LIBS})
  else()
      target_link_libraries(${test_name} PRIVATE ${VS_LIB_UNIT_TEST} ${VS_DISCOVERY} ${LIBS})
  endif()

  set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${test_name} PROPERTY FOLDER "lib/unit tests")

  target_include_directories(${test_name} PRIVATE ../)
  target_include_directories(${test_name} PRIVATE ${JSON_PARSER_ROOT_DIR})
  target_include_directories(${test_name} PRIVATE ${CMAKE_EXTERNAL_DEPS}/include ${OpenCV_INCLUDE_DIRS})
    # TODO: only needed for ukfQuaternion
  target_include_directories(${test_name} PRIVATE ${EIGEN3_INCLUDE_DIRS})
  target_include_directories(${test_name} PRIVATE ${TESTING_INCLUDE})
  target_compile_definitions(${test_name} PRIVATE VS_${GPU_BACKEND_DEFAULT})

  include_lib_vs_headers(${test_name})
  include_discovery_vs_headers(${test_name})
  target_link_libraries(${test_name} PRIVATE ${VS_DISCOVERY})

  foreach(vs_lib_sys_include ${VS_LIB_SYSTEM_INCLUDES})
    target_include_directories(${test_name} SYSTEM PRIVATE ${vs_lib_sys_include})
  endforeach()

  add_test(
      NAME ${test_name}
      COMMAND ${test_name} ${test_args}
      WORKING_DIRECTORY ${TEST_SRC_PATH}
      )
  set_tests_properties(${test_name} PROPERTIES LABELS "${test_labels}")
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "VS_TEST_DATA_DIR=${VS_TEST_DATA_DIR}")
endfunction()

if(STAGING)
  # XXX: ADD HERE NEW TESTS
else(STAGING)
  add_vs_unit_test(AffineTest affineTest.cpp)
  add_vs_unit_test(AmbisonicTest ambisonicTest.cpp)
  add_vs_unit_test(AudioDelayTest audioDelayTest.cpp)
  add_vs_unit_test(AudioFilterTest audioFilterTest.cpp)
  add_vs_unit_test(AudioSampleConvertTest audioSampleConvertTest.cpp)
  add_vs_unit_test(AudioToolsTest audioToolsTest.cpp)
  add_vs_unit_test(AutoCropTest autoCropTest.cpp)
  add_vs_unit_test(Base64Test base64Test.cpp)
  add_vs_unit_test(BijectiveTransformTest bijectiveTransformTest.cpp)
  add_vs_unit_test(BufferedReaderTest bufferedReaderTest.cpp)
  add_vs_unit_test(BufferTest bufferTest.cpp)
  add_vs_unit_test(CalibrationResetTest calibrationResetTest.cpp)
  add_vs_unit_test(CalibrationWithTranslationTest calibrationWithTranslationTest.cpp)
  add_vs_unit_test(ControllerInputFramesTest controllerInputFramesTest.cpp)
  add_vs_unit_test(ControlPointsConnectivityTest controlPointsConnectivityTest.cpp)
  add_vs_unit_test(CropAreaLensCenterTest cropAreaLensCenterTest.cpp)
  add_vs_unit_test(CurveTest curveTest.cpp)
  add_vs_unit_test(DrawNumberTest drawNumberTest.cpp)
  add_vs_unit_test(FrameRateTest frameRateTest.cpp)
  add_vs_unit_test(GridTest gridTest.cpp)
  add_vs_unit_test(HostTransformTest hostTransformTest.cpp)
  add_vs_unit_test(IMUStabilizationTest imuStabilizationTest.cpp)
  add_vs_unit_test(InputTest inputTest.cpp)
  add_vs_unit_test(LogTest logTest.cpp)
  add_vs_unit_test(MetadataExposureTest metadataExposureTest.cpp)
  add_vs_unit_test(MetadataToneCurveTest metadataToneCurveTest.cpp)
  add_vs_unit_test(MutexTest mutexTest.cpp)
  add_vs_unit_Test(OptimalPanoSizeTest optimalPanoSizeTest.cpp)
  add_vs_unit_test(OutputEventManagerTest outputEventManagerTest.cpp)
  add_vs_unit_test(PanoRigCompatibleTest panoRigCompatibleTest.cpp)
  add_vs_unit_test(PanoUpdaterTest panoUpdaterTest.cpp)
  add_vs_unit_test(ParsingTest parsingTest.cpp)
  add_vs_unit_test(ProceduralTest proceduralTest.cpp)
  add_vs_unit_test(PtsCommaParsingTest ptsCommaParsingTest.cpp)
  add_vs_unit_test(PtsParsingTest ptsParsingTest.cpp)
  add_vs_unit_test(PtvAudioPipeTest ptvAudioPipeTest.cpp)
  add_vs_unit_test(PtvDefaultsTest ptvDefaultsTest.cpp)
  add_vs_unit_test(PtvEquirectInputSerializeTest ptvEquirectInputSerializeTest.cpp)
  add_vs_unit_test(PtvTest ptvTest.cpp)
  add_vs_unit_test(ReaderControllerTest readerControllerTest.cpp)
  add_vs_unit_test(RectTest rectTest.cpp)
  add_vs_unit_test(RemoveInputTest removeInputTest.cpp)
  add_vs_unit_test(ResetInputGeometryTest resetInputGeometryTest.cpp)
  add_vs_unit_test(RigDefTest rigDefTest.cpp)
  add_vs_unit_test(RotationTest rotationTest.cpp)
  add_vs_unit_test(SimpleKernelTest simpleKernelTest.cpp)
  add_vs_unit_test(StabFilterTest stabFilterTest.cpp)
  add_vs_unit_test(StatusTest statusTest.cpp)
  add_vs_unit_test(StitchOutputTest stitchOutputTest.cpp)
  add_vs_unit_test(StreamTest streamTest.cpp)
  add_vs_unit_test(UndistortTest undistortTest.cpp)

  if(NOT (LINUX AND ${GPU_BACKEND_DEFAULT} STREQUAL OPENCL))
    add_vs_unit_test(MappingTest mappingTest.cpp)
  endif()

  if(NOT OCLGRIND)
    add_vs_unit_test(ControllerTest controllerTest.cpp)
  endif()

  # removes the cache dir which is used by other tests
  add_vs_test_arg(DirUtilsTest dirutilstest.cpp "cmd" "")

  #unit tests that doesn't link with the lib but compiles needed parts of it

  #bison and flex target files properties have to be redefined at this level
  # to indicate they are generated and to shut up warnings
  if(WINDOWS)
    set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES GENERATED TRUE CMAKE_CXX_FLAGS "-w")
  elseif(APPLE)
    set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES GENERATED TRUE COMPILE_FLAGS "-w -Xanalyzer -analyzer-disable-all-checks")
  else()
    set_source_files_properties(${BISON_jsonParser_OUTPUTS} ${FLEX_jsonLexer_OUTPUTS} PROPERTIES GENERATED TRUE COMPILE_FLAGS "-w")
  endif(WINDOWS)

  # loads large images, not fast enough to be unit test
  add_vs_test_arg(CompressionTest compressionTest.cpp "cmd" "")

  # can't run in parallel
  add_vs_test_arg(ReaderControllerAsyncTest readerControllerAsyncTest.cpp "cmd" "")

  if(NOT LINUX OR ${GPU_BACKEND_DEFAULT} STREQUAL CUDA)
    # VSA-7248
    add_vs_unit_test(UnpackTest unpackTest.cpp)
    add_vs_unit_test(VoronoiTest voronoiTest.cpp)
  endif()

  if((${GPU_BACKEND_DEFAULT} STREQUAL CUDA OR NOT WINDOWS) AND NOT OCLGRIND)
    add_vs_unit_test(SynchroTest synchroTest.cpp)
  endif()

  add_vs_unit_test(ThreadPoolTest threadPoolTest.cpp)
  add_vs_unit_test(TransformStackTest transformStackTest.cpp)
  add_vs_unit_test(UkfQuaternionTest ukfQuaternionTest.cpp)
  add_vs_unit_test(CircularBufferTest circularBufferTest.cpp)

  if(NOT APPLE)
    # this test is not working on the mac Nvidia buildbot
    # a known problem already in the Scons test
    add_vs_unit_test(UtilTest utilTest.cpp)
  endif(NOT APPLE)

  if(${GPU_BACKEND_DEFAULT} STREQUAL CUDA)
    #  add_vs_unit_test(BlurRGBA210Test blurRGBA210Test.cpp)
    add_vs_unit_test(BlurTest blurTest.cpp)
    add_vs_unit_test(DownsamplingTest downsamplingTest.cpp)
    add_vs_unit_test(FlowBasedBlendingTest flowBasedBlendingTest.cpp)
    add_vs_unit_test(HistogramTest histogramTest.cpp)
    add_vs_unit_test(PyramidTest pyramidTest.cpp)
    add_vs_unit_test(ReduceTest reduceTest.cpp)
    add_vs_unit_test(RenderTest renderTest.cpp)
    add_vs_unit_test(SamplingTest samplingTest.cpp)
    add_vs_unit_test(SeamTest seamTest.cpp)
    add_vs_unit_test(SpaceTransformTest spaceTransformTest.cpp)
    add_vs_unit_test(ProcessorsTest processorsTest.cpp)

    add_vs_test_arg(ContourMatchingTest contourMatchingTest.cpp "cmd" "" ${OpenCV_LIBRARIES})
    add_vs_test_arg(CoordTransformTest coordTransformTest.cpp "cmd" "")
    add_vs_test_arg(MaskInterpolationTest maskInterpolationTest.cpp "cmd" "")

  endif(${GPU_BACKEND_DEFAULT} STREQUAL CUDA)

  set(OrahAudioSyncTest_Example ${VS_ASSETS}/orah_audio/test.wav)
  if(EXISTS ${OrahAudioSyncTest_Example})
    add_vs_test_arg(OrahAudioSyncTest orahAudioSyncTest.cpp "unit" ${OrahAudioSyncTest_Example})
  endif()

endif(STAGING)
