# CMAKE_C_FLAGS_ASAN - Flags to use for C with asan
# CMAKE_CXX_FLAGS_ASAN  - Flags to use for C++ with asan
# HAVE_ADDRESS_SANITIZER - True or false if the ASan build type is available
#
# ---
#
# https://github.com/arsenm/sanitizers-cmake
# The MIT License (MIT)
#
# Copyright (c)
#   2013 Matthew Arsenault
#   2015-2016 RWTH Aachen University, Federal Republic of Germany
#

include(CheckCCompilerFlag)

# Set -Werror to catch "argument unused during compilation" warnings
set(CMAKE_REQUIRED_FLAGS "-Werror -faddress-sanitizer") # Also needs to be a link flag for test to pass
check_c_compiler_flag("-faddress-sanitizer" HAVE_FLAG_ADDRESS_SANITIZER)

set(CMAKE_REQUIRED_FLAGS "-Werror -fsanitize=address") # Also needs to be a link flag for test to pass
check_c_compiler_flag("-fsanitize=address" HAVE_FLAG_SANITIZE_ADDRESS)

unset(CMAKE_REQUIRED_FLAGS)

if(HAVE_FLAG_SANITIZE_ADDRESS)
  # Clang 3.2+ use this version
  set(ADDRESS_SANITIZER_FLAG "-fsanitize=address")
elseif(HAVE_FLAG_ADDRESS_SANITIZER)
  # Older deprecated flag for ASan
  set(ADDRESS_SANITIZER_FLAG "-faddress-sanitizer")
endif()

if(NOT ADDRESS_SANITIZER_FLAG)
  return()
else(NOT ADDRESS_SANITIZER_FLAG)
  set(HAVE_ADDRESS_SANITIZER FALSE)
endif()

set(HAVE_ADDRESS_SANITIZER TRUE)

set(CMAKE_C_FLAGS_ASAN "-O1 -g ${ADDRESS_SANITIZER_FLAG} -fno-omit-frame-pointer -fno-optimize-sibling-calls"
    CACHE STRING "Flags used by the C compiler during ASan builds."
    FORCE)
set(CMAKE_CXX_FLAGS_ASAN "-O1 -g ${ADDRESS_SANITIZER_FLAG} -fno-omit-frame-pointer -fno-optimize-sibling-calls"
    CACHE STRING "Flags used by the C++ compiler during ASan builds."
    FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${ADDRESS_SANITIZER_FLAG}"
    CACHE STRING "Flags used for linking binaries during ASan builds."
    FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN "${ADDRESS_SANITIZER_FLAG}"
    CACHE STRING "Flags used by the shared libraries linker during ASan builds."
    FORCE)
mark_as_advanced(CMAKE_C_FLAGS_ASAN
                 CMAKE_CXX_FLAGS_ASAN
                 CMAKE_EXE_LINKER_FLAGS_ASAN
                 CMAKE_SHARED_LINKER_FLAGS_ASAN)
