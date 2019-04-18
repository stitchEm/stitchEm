#
# https://github.com/arsenm/sanitizers-cmake
# The MIT License (MIT)
#
# Copyright (c)
#   2013 Matthew Arsenault
#   2015-2016 RWTH Aachen University, Federal Republic of Germany
#


include(CheckCXXCompilerFlag)

# Blacklist file for Undefined Behavior Sanitizer
# APPEND lines to this file to disable checks for source files or functions
# e.g. APPEND "src:/absolute_path_to_src/file.cpp"
#
# ccache may need to be disabled to force recompilation after changes to this file
set(UBSAN_BLACKLIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/ubsan_blacklist.txt)

if(COMPILER_CLANG)
  set(UBSAN_FLAGS "-fsanitize=undefined,integer -fno-sanitize-recover=undefined -fsanitize-blacklist=${UBSAN_BLACKLIST_FILE}")
elseif(COMPILER_GCC)
  set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=undefined -fsanitize-blacklist=${UBSAN_BLACKLIST_FILE}")
endif()

set(CMAKE_REQUIRED_FLAGS )
check_cxx_compiler_flag($UBSAN_FLAGS HAVE_FLAG_SANITIZE_UNDEFINED_BEHAVIOR)

unset(CMAKE_REQUIRED_FLAGS)

if(HAVE_FLAG_SANITIZE_UNDEFINED_BEHAVIOR)

  set(HAVE_UNDEFINED_BEHAVIOR_SANITIZER TRUE)

  set(CMAKE_C_FLAGS_UBSAN "-O1 -g -fno-omit-frame-pointer ${UBSAN_FLAGS}"
      CACHE STRING "Flags used by the C compiler during UBSan builds."
      FORCE)
  set(CMAKE_CXX_FLAGS_UBSAN "-O1 -g -fno-omit-frame-pointer ${UBSAN_FLAGS}"
      CACHE STRING "Flags used by the C++ compiler during UBSan builds."
      FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_UBSAN "-fno-omit-frame-pointer ${UBSAN_FLAGS}"
      CACHE STRING "Flags used for linking binaries during UBSan builds."
      FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_UBSAN "-fno-omit-frame-pointer ${UBSAN_FLAGS}"
      CACHE STRING "Flags used by the shared libraries linker during UBSan builds."
      FORCE)

  file(COPY ${CMAKE_SOURCE_DIR}/tests/ubsan_blacklist_template.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/ubsan_blacklist_template.txt ${UBSAN_BLACKLIST_FILE})

  mark_as_advanced(CMAKE_C_FLAGS_UBSAN
                   CMAKE_CXX_FLAGS_UBSAN
                   CMAKE_EXE_LINKER_FLAGS_UBSAN
                   CMAKE_SHARED_LINKER_FLAGS_UBSAN)
endif(HAVE_FLAG_SANITIZE_UNDEFINED_BEHAVIOR)
