# test the h264_nvenc in PTV

Feature: Output codec
    Scenario: Output codec
        When I launch videostitch-cmd with videoformat01/h264_nvenc.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed
        When I check videoformat01/output.mp4 integrity with avprobe
        Then The video is OK

    @slow
    Scenario Outline: Audio Video chunk alignment
        When I launch videostitch-cmd with <folder>/<ptv>_multifiles.ptv and "-d 0 -v 3 -l 495"
        Then I expect the command to succeed
        And  The file size of <folder>/output*.<fformat> is below <limit> bytes
        When I check files <folder>/output <fformat> streams alignment with avprobe
        When I check files <folder>/output*.<fformat> integrity with avprobe
        Then The videos are OK

        Examples:
            | folder        | ptv                             | fformat | limit    |
            | videoformat01 | template_mp4_h264_nvenc_audio   | mp4     | 15485760 |

