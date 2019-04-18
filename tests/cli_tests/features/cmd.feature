# A really simple test to see if tools used without
# argument are printing usage

@slow
Feature: cmd without arguments

    Scenario Outline: cmd without arguments
        When I launch <tool> without arguments
        Then I expect the command to fail with code 1

        Examples:
            | tool              |
            | videostitch-cmd   |
            | calibrationimport |

