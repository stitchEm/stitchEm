# CMAKE_C_FLAGS_LSAN - Flags to use for C with lsan
# CMAKE_CXX_FLAGS_LSAN  - Flags to use for C++ with lsan
# HAVE_LEAK_SANITIZER - True or false if the ASan build type is available
#
# https://github.com/arsenm/sanitizers-cmake
# The MIT License (MIT)
#
# Copyright (c)
#   2013 Matthew Arsenault
#   2015-2016 RWTH Aachen University, Federal Republic of Germany
#


include(CheckCCompilerFlag)

set(LEAK_SANITIZER_FLAG -fsanitize=leak)

# Set -Werror to catch "argument unused during compilation" warnings
set(CMAKE_REQUIRED_FLAGS "-Werror ${LEAK_SANITIZER_FLAG}") # Also needs to be a link flag for test to pass
check_c_compiler_flag("${LEAK_SANITIZER_FLAG}" HAVE_FLAG_LEAK_SANITIZER)

unset(CMAKE_REQUIRED_FLAGS)

if(NOT HAVE_FLAG_LEAK_SANITIZER)
  set(HAVE_LEAK_SANITIZER NO)
  return()
endif(NOT HAVE_FLAG_LEAK_SANITIZER)

set(HAVE_LEAK_SANITIZER YES)

set(CMAKE_C_FLAGS_LSAN "-O1 -g ${LEAK_SANITIZER_FLAG} -fno-omit-frame-pointer -fno-optimize-sibling-calls"
    CACHE STRING "Flags used by the C compiler during LSan builds."
    FORCE)
set(CMAKE_CXX_FLAGS_LSAN "-O1 -g ${LEAK_SANITIZER_FLAG} -fno-omit-frame-pointer -fno-optimize-sibling-calls"
    CACHE STRING "Flags used by the C++ compiler during LSan builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_LSAN "${LEAK_SANITIZER_FLAG}"
    CACHE STRING "Flags used for linking binaries during LSan builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_LSAN "${LEAK_SANITIZER_FLAG}"
    CACHE STRING "Flags used by the shared libraries linker during LSan builds."
    FORCE)
mark_as_advanced(CMAKE_C_FLAGS_LSAN
                 CMAKE_CXX_FLAGS_LSAN
                 CMAKE_EXE_LINKER_FLAGS_LSAN
                 CMAKE_SHARED_LINKER_FLAGS_LSAN)
