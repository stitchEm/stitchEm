#
# https://github.com/arsenm/sanitizers-cmake
# The MIT License (MIT)
#
# Copyright (c)
#   2013 Matthew Arsenault
#   2015-2016 RWTH Aachen University, Federal Republic of Germany
#

include(CheckCCompilerFlag)

set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
check_c_compiler_flag("-fsanitize=thread" HAVE_FLAG_SANITIZE_THREAD)

unset(CMAKE_REQUIRED_FLAGS)

if(HAVE_FLAG_SANITIZE_THREAD)

  set(HAVE_THREAD_SANITIZER TRUE)

  set(CMAKE_C_FLAGS_TSAN "-O2 -g -fsanitize=thread"
      CACHE STRING "Flags used by the C compiler during TSan builds."
      FORCE)
  set(CMAKE_CXX_FLAGS_TSAN "-O2 -g -fsanitize=thread"
      CACHE STRING "Flags used by the C++ compiler during TSan builds."
      FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread"
      CACHE STRING "Flags used for linking binaries during TSan builds."
      FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_TSAN "-fsanitize=thread"
      CACHE STRING "Flags used by the shared libraries linker during TSan builds."
      FORCE)

  mark_as_advanced(CMAKE_C_FLAGS_TSAN
                   CMAKE_CXX_FLAGS_TSAN
                   CMAKE_EXE_LINKER_FLAGS_TSAN
                   CMAKE_SHARED_LINKER_FLAGS_TSAN)
endif(HAVE_FLAG_SANITIZE_THREAD)
