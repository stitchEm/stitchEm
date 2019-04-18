set(HAVE_DISPLAY YES)

if(LINUX)
  execute_process(COMMAND xdpyinfo
                  RESULT_VARIABLE XDPYINFO_EXIT_CODE
                  OUTPUT_QUIET ERROR_QUIET)
  if(XDPYINFO_EXIT_CODE GREATER 0)
    # TODO LINUX: only disable the tests that really need a display
    # or fake a display?
    set(HAVE_DISPLAY NO)
  endif(XDPYINFO_EXIT_CODE GREATER 0)
endif(LINUX)