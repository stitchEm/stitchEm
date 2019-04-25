PROJECT(appsTest)

include(EnableXServer)

# ----------------------------------------------------------------------------
# Output directories
# We need to be in the output directory on windows
# ----------------------------------------------------------------------------

if(NOT WINDOWS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VS_TEST_DIR})
endif(NOT WINDOWS)

if(CMAKE_BUILD_TYPE)
  # CMAKE-TODO: changes to this folder not updated on reconfiguration
  file(COPY "data" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UP)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP} ${VS_TEST_DIR_${OUTPUTCONFIG_UP}})

    # CMAKE-TODO: changes to this folder not updated on reconfiguration
    file(COPY "data" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UP}}")
  endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)
endif(CMAKE_BUILD_TYPE)

# ----------------------------------------------------------------------------

find_package(Qt5Test REQUIRED HINTS ${qt_dir})

function(add_qt_ui_test test_name sources ui_files)
  qt5_wrap_ui(UI_GENERATED_HEADERS "${ui_files}")
  set(SOURCES
      "${test_name}/${test_name}.cpp"
      "${sources}"
      "${UI_GENERATED_HEADERS}"
      )
  add_qt_unit_test_internal(${test_name} "${SOURCES}")
endfunction()

function(add_qt_unit_test test_name)
  add_qt_unit_test_internal(${test_name} "${test_name}/${test_name}.cpp")
endfunction()

function(add_qt_unit_test_internal test_name sources)
  add_executable(${test_name} ${sources})
  set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
  set_property(TARGET ${test_name} PROPERTY FOLDER "apps/tests")
  include_lib_vs_headers(${test_name})
  include_discovery_vs_headers(${test_name})
  target_link_libraries(${test_name} PRIVATE ${VS_GUI} ${VS_BASE} Qt5::Test Qt5::Concurrent Qt5::Gui Qt5::Multimedia Qt5::OpenGL Qt5::Widgets)
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set_tests_properties(${test_name} PROPERTIES LABELS "apps")
  if (WINDOWS)
    # Aweful cmake bug https://cmake.org/pipermail/cmake/2010-December/041176.html
    set(PATH_STRING "${QT_PATH};$ENV{PATH}")
    string(REPLACE ";" "\\;" PATH_STRING "${PATH_STRING}")
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PATH=${PATH_STRING}")
  endif(WINDOWS)
endfunction()

if(HAVE_DISPLAY)
  if(STAGING)
    set(TESTS
        # XXX: ADD HERE NEW TESTS
        )
  else(STAGING)
    add_qt_unit_test(audioHelperTest)
    add_qt_unit_test(cropInputTest)
    add_qt_unit_test(extensionsTest)
    add_qt_unit_test(panoSizeTest)
    add_qt_unit_test(smartEnumTest)
    add_qt_unit_test(stitchingWindowTest)
    add_qt_unit_test(templateTest)
    add_qt_unit_test(timeConverterTest)
    add_qt_unit_test(updateCheckerTest)

    add_qt_unit_test(normalizeInputTest)
    set_tests_properties(normalizeInputTest PROPERTIES LABELS "apps gpu")

    link_target_to_libvideostitch(audioHelperTest)
    link_target_to_libvideostitch(normalizeInputTest)
    target_link_libraries(normalizeInputTest PRIVATE ${VS_DISCOVERY})
    target_link_libraries(stitchingWindowTest PRIVATE ${VS_DISCOVERY})

  endif(STAGING)
endif(HAVE_DISPLAY)

