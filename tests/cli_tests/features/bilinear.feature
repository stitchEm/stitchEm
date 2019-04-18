# Test to see ensure the inputs are accessed with bilinear interpolation

Feature: Bilinear interpolation

    Scenario Outline: Bilinear interpolation
        When I launch videostitch-cmd with imageQuality/<test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare imageQuality/<test>-0.jpg with imageQuality/reference-<test>.jpg
        Then I expect the comparison error to be less than 0.005

    Examples:
            | test              |
            | cross             |
            | cross-inv         |
            | cross-orientation |
