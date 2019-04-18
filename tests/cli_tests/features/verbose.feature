# Testing different verbose levels are working

@slow
Feature: Verbosity levels

    Scenario Outline: Verbosity levels
        When I launch videostitch-cmd with procedural/checker.ptv and "-d 0 -v <n> -f 0 -l 10"
        Then I expect the command to succeed

        Examples:
            | n |
            | 0 |
            | 1 |
            | 2 |
            | 3 |
            | q |

