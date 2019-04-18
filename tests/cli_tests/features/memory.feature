# Testing a really large project to see how cmd-line behaves

Feature: test big project

    Scenario: Out of memory
        When I launch videostitch-cmd with videoformat01/full_40000.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to fail with code 1

    Scenario: Pano size > 2 GB: 50k x 20k x 4 = 0x12A05F200, which is > 0x7fffffff
        When I launch videostitch-cmd with videoformat01/full_50000.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to fail with code 1

