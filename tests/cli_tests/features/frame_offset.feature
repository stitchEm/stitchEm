# Test to process a ptv with frame offset

Feature: ptv with frame offset

    Scenario: ptv with frame offset
        When I launch videostitch-cmd with videoformat01/frame_offset.ptv and "-d 0 -v 3 -f 0 -l 10"
        Then I expect the command to succeed

