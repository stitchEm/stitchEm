add_custom_target(
    astyle
    COMMAND
        astyle
        -r
        --exclude=lib/bindings
        --dry-run
        --indent=spaces=2
        --style=attach
        --keep-one-line-statements
        --add-brackets
        *.hpp *.h *.hxx *.cpp *.cu *.cl *.gpu
        > "${CMAKE_CURRENT_BINARY_DIR}/astyle1.tmp"
    COMMAND
         cat "${CMAKE_CURRENT_BINARY_DIR}/astyle1.tmp" | grep "Formatted" | wc -l
         > "${CMAKE_CURRENT_BINARY_DIR}/astyle2.tmp"
    COMMAND
        if [ `cat ${CMAKE_CURRENT_BINARY_DIR}/astyle2.tmp` -eq "0" ]; then exit "0" \; fi \; exit "1"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

