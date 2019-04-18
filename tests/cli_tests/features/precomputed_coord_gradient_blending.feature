# Test to see behaviour of gradient blending with precomputed coordinate buffers

Feature: Gradient blending

    Scenario Outline: Gradient blending
        When I launch videostitch-cmd with <test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare <test>-out-0.jpg with <test>-Reference-out-1.jpg
        Then I expect the comparison error to be less than 0.005

    Examples:
            | test                                  |
            | redbull/precomputed_coordinate_buffer |
