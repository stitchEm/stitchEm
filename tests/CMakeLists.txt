option(CLI_TESTS "Launch videostitch-cmd tests" ON)

if(CLI_TESTS)
  add_subdirectory(cli_tests)
endif(CLI_TESTS)

