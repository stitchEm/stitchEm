# Testing different procedurals ptvs

Feature: procedurals

    Scenario Outline: procedurals
        When I launch videostitch-cmd with procedural/<ptv> and "-d 0 -v 3 -f 1 -l 10"
        Then I expect the command to succeed

        Examples:
            | ptv         |
            | checker.ptv |
            | color.ptv   |
            | number.ptv  |

    # Disabled until VSA-6935 is fixed
    @wip
    Scenario: Checkerboard comparison
        When I launch videostitch-cmd with procedural/checker-colour.ptv and "-d 0 -v 3 -f 0 -l 1"
        Then I expect the command to succeed
        When I compare procedural/output-checker-colour-1.png with procedural/checker-colour-reference.png
        Then I expect the comparison error to be less than 0.03
