### Options ###
# XXX Only 44.1 kHz audio is currently
# implemented, since this is what the
# camera provides.
# Value can be 44 or 48.
set(SAMPLE_RATE_ 44)

### Target names ###
set(O2B_LIB o2b)

### Library ###
set(O2B_LIB_SOURCE_FILES
    src/fftsg.c
    src/orah4i2b.cpp)

set(O2B_LIB_HEADER_FILES
    header/orah4i2b.hpp)

add_library(${O2B_LIB} STATIC ${O2B_LIB_SOURCE_FILES})

### Includes ###
include_directories("${CMAKE_CURRENT_LIST_DIR}/src")
include_directories("${CMAKE_CURRENT_LIST_DIR}/header")

### Libraries ###
if (LINUX)
    target_link_libraries(${O2B_LIB} "m")
endif()

### Set Flags ###
if (WINDOWS)
    target_compile_definitions(${O2B_LIB} PRIVATE _USE_MATH_DEFINES)
elseif (APPLE)
    set_property(TARGET ${O2B_LIB} APPEND_STRING PROPERTY COMPILE_FLAGS " -ffast-math")
elseif (LINUX)
    set_target_properties(${O2B_LIB} PROPERTIES
      COMPILE_FLAGS " -ffast-math -fPIC")
endif()
target_compile_definitions(${O2B_LIB} PRIVATE SAMPLE_RATE=${SAMPLE_RATE_})

### Tests ###
set(O2B_TEST Orah4iToBFormat)
add_executable(${O2B_TEST} src/orah4i2b_test.cpp)
target_link_libraries(${O2B_TEST} ${O2B_LIB})
if(STAGING)
  # XXX: ADD HERE NEW TESTS
else(STAGING)
  add_test(NAME ${O2B_TEST} COMMAND ${O2B_TEST})
endif(STAGING)
