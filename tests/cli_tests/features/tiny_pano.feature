Feature: Stitching small panorama

    Scenario Outline: Stitching 64x32 px panorama
        When I launch videostitch-cmd with videoformat01/<test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare videoformat01/<test>-0.png with videoformat01/reference-<test>-1.png
        Then I expect the comparison error to be less than 0.005

    Examples:
            | test        |
            | gradient_64 |
