# Test to see behaviour of Laplacian blending

Feature: Laplacian blending

    Scenario Outline: Laplacian blending
        When I launch videostitch-cmd with laplacianBlending/<test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare laplacianBlending/<test>-out-0.jpg with laplacianBlending/Reference<test>-out-0.jpg
        Then I expect the comparison error to be less than 0.03

		Examples:
            | test                    |
            | PrecomputedCoordRafting |

    Scenario Outline: Stitching identity accuracy
        When I launch videostitch-cmd with input_formats/<test>-<format>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare input_formats/<format>-out-0.<format> with input_formats/input.<format>
        Then I expect the comparison error to be less than <expected_error>

		Examples:
            | test               | format | expected_error |
            | identity-precomp   | png    | 0.012          |
            | identity-preshrink | png    | 0.015          |
