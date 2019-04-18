# - try to find cppcheck tool
#
# Cache Variables:
#  CPPCHECK_EXECUTABLE
#
# Non-cache variables you might use in your CMakeLists.txt:
#  CPPCHECK_FOUND
#  CPPCHECK_VERSION
#  CPPCHECK_POSSIBLEERROR_ARG
#  CPPCHECK_UNUSEDFUNC_ARG
#  CPPCHECK_STYLE_ARG
#  CPPCHECK_WARNING_ARG
#  CPPCHECK_PERF_ARG
#  CPPCHECK_PORTABILITY_ARG
#  CPPCHECK_QUIET_ARG
#  CPPCHECK_INCLUDEPATH_ARG
#  CPPCHECK_JOBS_ARG
#  CPPCHECK_ERROR_EXIT_CODE_ARG
#  CPPCHECK_MARK_AS_ADVANCED - whether to mark our vars as advanced even
#    if we don't find this program.
#
# Simplification of
# https://github.com/rpavlik/cmake-modules/blob/master/Findcppcheck.cmake

file(TO_CMAKE_PATH "${CPPCHECK_ROOT_DIR}" CPPCHECK_ROOT_DIR)
set(CPPCHECK_ROOT_DIR
    "${CPPCHECK_ROOT_DIR}"
    CACHE
    PATH
    "Path to search for cppcheck")

# cppcheck app bundles on Mac OS X are GUI, we want command line only
set(_oldappbundlesetting ${CMAKE_FIND_APPBUNDLE})
set(CMAKE_FIND_APPBUNDLE NEVER)

if(CPPCHECK_EXECUTABLE AND NOT EXISTS "${CPPCHECK_EXECUTABLE}")
    set(CPPCHECK_EXECUTABLE "notfound" CACHE PATH FORCE "")
endif()

# If we have a custom path, look there first.
if(CPPCHECK_ROOT_DIR)
    find_program(CPPCHECK_EXECUTABLE
        NAMES
        cppcheck
        cli
        PATHS
        "${CPPCHECK_ROOT_DIR}"
        PATH_SUFFIXES
        cli
        NO_DEFAULT_PATH)
endif()

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)

# Restore original setting for appbundle finding
set(CMAKE_FIND_APPBUNDLE ${_oldappbundlesetting})

# Find out where our test file is
get_filename_component(_cppcheckmoddir ${CMAKE_CURRENT_LIST_FILE} PATH)
set(_cppcheckdummyfile "${_cppcheckmoddir}/Findcppcheck.cpp")
if(NOT EXISTS "${_cppcheckdummyfile}")
    message(FATAL_ERROR
        "Missing file ${_cppcheckdummyfile} - should be alongside Findcppcheck.cmake, can be found at https://github.com/rpavlik/cmake-modules")
endif()

function(_cppcheck_test_arg _resultvar _arg)
    if(NOT CPPCHECK_EXECUTABLE)
        set(${_resultvar} NO)
        return()
    endif()
    execute_process(COMMAND
        "${CPPCHECK_EXECUTABLE}"
        "${_arg}"
        "--quiet"
        "${_cppcheckdummyfile}"
        RESULT_VARIABLE
        _cppcheck_result
        OUTPUT_QUIET
        ERROR_QUIET)
    if("${_cppcheck_result}" EQUAL 0)
        set(${_resultvar} YES PARENT_SCOPE)
    else()
        set(${_resultvar} NO PARENT_SCOPE)
    endif()
endfunction()

function(_cppcheck_set_arg_var _argvar _arg)
    if("${${_argvar}}" STREQUAL "")
        _cppcheck_test_arg(_cppcheck_arg "${_arg}")
        if(_cppcheck_arg)
            set(${_argvar} "${_arg}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

if(CPPCHECK_EXECUTABLE)

    set(CPPCHECK_STYLE_ARG "--enable=style")
    set(CPPCHECK_UNUSEDFUNC_ARG "--enable=unusedFunction")
    set(CPPCHECK_INFORMATION_ARG "--enable=information")
    set(CPPCHECK_MISSINGINCLUDE_ARG "--enable=missingInclude")
    set(CPPCHECK_POSIX_ARG "--enable=posix")
    set(CPPCHECK_WARNING_ARG "--enable=warning")
    set(CPPCHECK_PERF_ARG "--enable=performance")
    set(CPPCHECK_PORTABILITY_ARG "--enable=portability")
    set(CPPCHECK_POSSIBLEERROR_ARG "--enable=all")
    set(CPPCHECK_QUIET_ARG "--quiet")
    set(CPPCHECK_INCLUDEPATH_ARG "-I")
    set(CPPCHECK_JOBS_ARG "-j")
    set(CPPCHECK_ERROR_EXIT_CODE_ARG "--error-exitcode=")

endif()

set(CPPCHECK_ALL
    "${CPPCHECK_EXECUTABLE} ${CPPCHECK_POSSIBLEERROR_ARG} ${CPPCHECK_UNUSEDFUNC_ARG} ${CPPCHECK_STYLE_ARG} ${CPPCHECK_QUIET_ARG} ${CPPCHECK_INCLUDEPATH_ARG} some/include/path")

execute_process(COMMAND "${CPPCHECK_EXECUTABLE}" --version
  OUTPUT_VARIABLE CPPCHECK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE ".* ([0-9]\\.([0-9]\\.[0-9])?)" "\\1"
    CPPCHECK_VERSION "${CPPCHECK_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cppcheck
    DEFAULT_MSG
    CPPCHECK_ALL
    CPPCHECK_EXECUTABLE
    CPPCHECK_POSSIBLEERROR_ARG
    CPPCHECK_UNUSEDFUNC_ARG
    CPPCHECK_STYLE_ARG
    CPPCHECK_INCLUDEPATH_ARG
    CPPCHECK_QUIET_ARG)

if(CPPCHECK_FOUND OR CPPCHECK_MARK_AS_ADVANCED)
    mark_as_advanced(CPPCHECK_ROOT_DIR)
endif()

mark_as_advanced(CPPCHECK_EXECUTABLE)
