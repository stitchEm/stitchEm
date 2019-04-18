function(get_git_version_tag APP)
  execute_process(COMMAND git describe --tags --match "${APP}-v?.*"
                  RESULT_VARIABLE git_exit_code
                  OUTPUT_VARIABLE version_from_git
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  ERROR_VARIABLE error_version_from_git)

  if(${git_exit_code})
    message("stderr from git: " ${error_version_from_git})
    message(FATAL_ERROR "git apps version check for ${APP} exited with code: " ${git_exit_code})
  else(${git_exit_code})
    string(STRIP ${version_from_git} version_from_git)
    set(${APP}_version_from_git ${version_from_git} PARENT_SCOPE)
  endif(${git_exit_code})
endfunction(get_git_version_tag)

function(get_git_branch)
  execute_process(COMMAND git rev-parse --abbrev-ref HEAD
                  RESULT_VARIABLE git_exit_code
                  OUTPUT_VARIABLE git_branch
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  ERROR_VARIABLE error_version_from_git)

  if(${git_exit_code})
    message("stderr from git: " ${error_version_from_git})
    message(FATAL_ERROR "git branch get exited with code: " ${git_exit_code})
  else(${git_exit_code})
    string(STRIP ${git_branch} git_branch)
    set(git_branch ${git_branch} PARENT_SCOPE)
  endif(${git_exit_code})
endfunction(get_git_branch)

function(get_git_timestamp)
    execute_process(COMMAND git show -s --format=%ad --date=short
                    RESULT_VARIABLE git_exit_code
                    OUTPUT_VARIABLE tag_timestamp
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    ERROR_VARIABLE error_version_from_git)

  if(${git_exit_code})
    message("stderr from git: " ${error_version_from_git})
    message(FATAL_ERROR "git log exited with code: " ${git_exit_code})
  else(${git_exit_code})
    string(STRIP ${tag_timestamp} tag_timestamp)
    set(tag_timestamp ${tag_timestamp} PARENT_SCOPE)
  endif(${git_exit_code})
endfunction(get_git_timestamp)

get_git_version_tag(Studio)
get_git_version_tag(VahanaVR)
get_git_version_tag(Player)
get_git_branch()
get_git_timestamp()

set(STUDIO_VERSION "${Studio_version_from_git}-${git_branch}.${tag_timestamp}")
set(VAHANA_VERSION "${VahanaVR_version_from_git}-${git_branch}.${tag_timestamp}")
set(PLAYER_VERSION "${Player_version_from_git}-${git_branch}.${tag_timestamp}")
message(STATUS "Studio version: " ${STUDIO_VERSION})

