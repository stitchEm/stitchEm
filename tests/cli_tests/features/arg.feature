# A really simple test to verify arguments order does not
# impact output

Feature: argument order test

    Scenario Outline: Argument order
        When I launch videostitch-cmd with arg/template.ptv and "<args>"
        Then I expect the command to succeed

        Examples:
            | args                    |
            | -v 4 -d 0 -f 0 -l 10    |
            | -d 0 -v 4 -f 0 -l 10    |

