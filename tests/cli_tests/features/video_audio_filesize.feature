Feature: Testing different video formats are working correctly with output file size limitations.

    @slow
    Scenario Outline: Audio Video chunk alignment and filesize
        When I launch videostitch-cmd with <folder>/<ptv>_multifiles.ptv and "-d 0 -v 3 -l 295"
        Then I expect the command to succeed
        And  The file size of <folder>/output*.<fformat> is below <limit> bytes
        When I check files <folder>/output <fformat> streams alignment with avprobe
        When I check files <folder>/output*.<fformat> integrity with avprobe
        Then The videos are OK

        # TODO: reduce limit with VSA-7422
        Examples:
            | folder        | ptv                             | fformat | limit    |
            | videoformat01 | template_mp4_mpeg4_audio        | mp4     | 15485760 |
            | videoformat01 | template_mov_h264_audio         | mov     | 29002928 |


