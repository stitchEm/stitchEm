# Note: this cmake script runs at build time (not at configure time).
# 
# The git version could have changed without CMake noticing,
# so it runs at every build

function(get_git_version_tag)
  execute_process(COMMAND git describe --tags --match "v?.*"
                  WORKING_DIRECTORY ${WORKING_DIR}
                  RESULT_VARIABLE git_exit_code
                  OUTPUT_VARIABLE version_from_git
                  ERROR_VARIABLE error_version_from_git)

  if(${git_exit_code})
    message("stderr from git: " ${error_version_from_git})
    message(FATAL_ERROR "git version check exited with code: " ${git_exit_code})
  else(${git_exit_code})
    string(STRIP ${version_from_git} version_from_git)
    set(version_from_git ${version_from_git} PARENT_SCOPE)
  endif(${git_exit_code})
endfunction(get_git_version_tag)

function(get_git_branch)
  execute_process(COMMAND git rev-parse --abbrev-ref HEAD
                  WORKING_DIRECTORY ${WORKING_DIR}
                  RESULT_VARIABLE git_exit_code
                  OUTPUT_VARIABLE git_branch
                  ERROR_VARIABLE error_version_from_git)

  if(${git_exit_code})
    message("stderr from git: " ${error_version_from_git})
    message(FATAL_ERROR "git branch get exited with code: " ${git_exit_code})
  else(${git_exit_code})
    string(STRIP ${git_branch} git_branch)
    set(git_branch ${git_branch} PARENT_SCOPE)
  endif(${git_exit_code})
endfunction(get_git_branch)

function(parse_lib_version)
  string(REGEX REPLACE
         "v([0-9]+)\\.([0-9]+)\\.([0-9]+).*"
         "\\1;\\2;\\3" version_list ${version_from_git})
  list(LENGTH version_list version_list_length)

  if(NOT ${version_list_length} EQUAL 3)
    message(FATAL_ERROR "Could not parse git tag into lib version")
  endif(NOT ${version_list_length} EQUAL 3)

  list(GET version_list 0 LIB_MAJOR)
  list(GET version_list 1 LIB_MINOR)
  list(GET version_list 2 LIB_REVISION)

  set(LIB_MAJOR ${LIB_MAJOR} PARENT_SCOPE)
  set(LIB_MINOR ${LIB_MINOR} PARENT_SCOPE)
  set(LIB_REVISION ${LIB_REVISION} PARENT_SCOPE)

endfunction(parse_lib_version)

get_git_version_tag()

get_git_branch()

parse_lib_version()
# message("lib version: v" ${LIB_MAJOR} "." ${LIB_MINOR} "." ${LIB_REVISION})

message("lib " ${version_from_git} " @ " ${git_branch})

# TODO doc/Doxyfile-library stuff

file(READ ${VERSION_TEMPLATE} version_file)

# CMake regex engine doesn't seem to understand ^ and $
# let's match ///...///\n which works for the template file
string(REGEX REPLACE
       "///.*///\r?\n"
       "" version_file ${version_file})

string(REPLACE "@@lib_version@@"
       ${version_from_git} version_file
       ${version_file})

string(REPLACE "@@lib_major@@"
       ${LIB_MAJOR} version_file
       ${version_file})

string(REPLACE "@@lib_minor@@"
       ${LIB_MINOR} version_file
       ${version_file})

string(REPLACE "@@lib_revision@@"
       ${LIB_REVISION} version_file
       ${version_file})

string(REPLACE "@@branch@@"
       ${git_branch} version_file
       ${version_file})

if(EXISTS ${VERSION_HEADER})
  file(READ ${VERSION_HEADER} old_version_file)
  string(COMPARE EQUAL ${version_file} ${old_version_file} version_file_unchanged)
else()
  set(version_file_unchanged NO)
endif(EXISTS ${VERSION_HEADER})

if(NOT version_file_unchanged)
  message("Writing version.hpp")
  file(WRITE ${VERSION_HEADER} ${version_file})
endif(NOT version_file_unchanged)

