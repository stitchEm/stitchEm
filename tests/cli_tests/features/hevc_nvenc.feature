# test the hevc_nvenc in PTV

Feature: Output codec
    @wip
    # FIXME : Restore test when new debian_wheezy_64 GPU is upgraded
    Scenario: Output codec
        When I launch videostitch-cmd with videoformat01/hevc_nvenc.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed
        When I check videoformat01/output.mp4 integrity with avprobe
        Then The video is OK
