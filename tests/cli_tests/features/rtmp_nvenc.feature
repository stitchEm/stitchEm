# test the RTMP streaming

Feature: RTMP streaming
    Scenario: RTMP streaming
        When I launch videostitch-cmd with videoformat01/rtmp_nvenc.ptv and "-d 0 -f 0 -l 100"
        Then I expect the command to succeed
