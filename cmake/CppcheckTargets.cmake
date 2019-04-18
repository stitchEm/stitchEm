# - Run cppcheck on c++ source files as a custom target
# Simplification of
# https://github.com/rpavlik/cmake-modules/blob/master/CppcheckTargets.cmake

if(__add_cppcheck)
    return()
endif()
set(__add_cppcheck YES)

if(NOT CPPCHECK_FOUND)
    find_package(cppcheck QUIET)
endif()

if(NOT CPPCHECK_FOUND)
    add_custom_target(all_cppcheck
        COMMENT "cppcheck executable not found")
    set_target_properties(all_cppcheck PROPERTIES EXCLUDE_FROM_ALL TRUE)
elseif(CPPCHECK_VERSION VERSION_LESS 1.53.0)
    add_custom_target(all_cppcheck
        COMMENT "Need at least cppcheck 1.53, found ${CPPCHECK_VERSION}")
    set_target_properties(all_cppcheck PROPERTIES EXCLUDE_FROM_ALL TRUE)
    set(CPPCHECK_FOUND)
endif()

if(NOT TARGET all_cppcheck)
  add_custom_target(all_cppcheck)
endif()

set(CPPCHECK_FOLDER "${PROJECT_SOURCE_DIR}/cppcheck")
set(CPP_SUPP_FILE ${CPPCHECK_FOLDER}/cppcheck-supp.txt)
set(CPPCHECK_XML "${CPPCHECK_FOLDER}/xml")
file(MAKE_DIRECTORY ${CPPCHECK_XML})

add_custom_target(cppcheck_xml_report
    COMMAND
    "./merge_xml.py"
    WORKING_DIRECTORY
    "${CPPCHECK_FOLDER}"
    COMMENT
    "creating XML report"
    VERBATIM)
add_custom_target(cppcheck_html_report
    COMMAND
    "cppcheck-htmlreport"
    "--file=report.xml"
    "--report-dir=html"
    "--source-dir=.."
    WORKING_DIRECTORY
    "${CPPCHECK_FOLDER}"
    COMMENT
    "creating HTML report"
    VERBATIM)
add_custom_target(cppcheck_exit
    COMMAND
    "./analyse_xml.py"
    WORKING_DIRECTORY
    "${CPPCHECK_FOLDER}"
    COMMENT
    "analyzing XML report"
    VERBATIM)

add_dependencies(cppcheck_html_report cppcheck_xml_report)
add_dependencies(cppcheck_exit cppcheck_html_report)
add_dependencies(all_cppcheck cppcheck_exit)

function(add_cppcheck _name)
    if(NOT TARGET ${_name})
        message(FATAL_ERROR
            "add_cppcheck given a target name that does not exist: '${_name}' !")
    endif()
    if(CPPCHECK_FOUND)
        set(_cppcheck_args -I ${CMAKE_SOURCE_DIR} ${CPPCHECK_EXTRA_ARGS})

        list(FIND ARGN UNUSED_FUNCTIONS _unused_func)
        if("${_unused_func}" GREATER "-1")
            list(APPEND _cppcheck_args ${CPPCHECK_UNUSEDFUNC_ARG})
        endif()

        list(FIND ARGN STYLE _style)
        if("${_style}" GREATER "-1")
            list(APPEND _cppcheck_args ${CPPCHECK_STYLE_ARG})
        endif()

        list(FIND ARGN POSSIBLE_ERROR _poss_err)
        if("${_poss_err}" GREATER "-1")
            list(APPEND _cppcheck_args ${CPPCHECK_POSSIBLEERROR_ARG})
        endif()

        list(FIND ARGN FORCE _force)
        if("${_force}" GREATER "-1")
            list(APPEND _cppcheck_args "--force")
        endif()

        list(FIND ARGN VS _vs)
        if("${_vs}" GREATER "-1")
            list(APPEND _cppcheck_args "--force")
            list(APPEND _cppcheck_args ${CPPCHECK_STYLE_ARG})
            list(APPEND _cppcheck_args ${CPPCHECK_INFORMATION_ARG})
            list(APPEND _cppcheck_args ${CPPCHECK_MISSINGINCLUDE_ARG})
            list(APPEND _cppcheck_args ${CPPCHECK_WARNING_ARG})
            list(APPEND _cppcheck_args ${CPPCHECK_PERF_ARG})
            list(APPEND _cppcheck_args ${CPPCHECK_PORTABILITY_ARG})
            list(APPEND _cppcheck_args "${CPPCHECK_JOBS_ARG}8")
            list(APPEND _cppcheck_args "--std=c++11")
            list(APPEND _cppcheck_args "--suppressions-list=${CPP_SUPP_FILE}")
            list(APPEND _cppcheck_args "--suppress=unmatchedSuppression")
        endif()

        get_target_property(_cppcheck_includes "${_name}" INCLUDE_DIRECTORIES)
        set(_includes)
        foreach(_include ${_cppcheck_includes})
            list(APPEND _includes "-I${_include}")
        endforeach()

        get_target_property(_cppcheck_sources "${_name}" SOURCES)
        set(_files)
        foreach(_source ${_cppcheck_sources})
            if(NOT "${_source}" MATCHES ".*TARGET_OBJECTS.*")
                get_source_file_property(_cppcheck_lang "${_source}" LANGUAGE)
                get_source_file_property(_cppcheck_loc "${_source}" LOCATION)
                if(("${_cppcheck_lang}" STREQUAL "C") OR ("${_cppcheck_lang}" STREQUAL "CXX"))
                    list(APPEND _files "${_cppcheck_loc}")
                endif()
            endif()
        endforeach()

        add_custom_target(${_name}_cppcheck
            COMMAND
            ${CPPCHECK_EXECUTABLE}
            ${CPPCHECK_QUIET_ARG}
            ${CPPCHECK_TEMPLATE_ARG}
            ${_cppcheck_args}
            ${_includes}
            ${_files}
            "--xml"
            "--xml-version=2"
            "2>" "${CPPCHECK_XML}/${_name}.xml"
            WORKING_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT
            "${_name}_cppcheck: Running cppcheck on target ${_name}..."
            VERBATIM)
        add_dependencies(all_cppcheck ${_name}_cppcheck)
        add_dependencies(cppcheck_xml_report ${_name}_cppcheck)
    endif()

endfunction()

