# Test to process ptv with output file size limitations

Feature: Testing different video formats are working correctly with output file size limitations

    @slow
    Scenario Outline: Video formats filesize
        When I launch videostitch-cmd with <folder>/<ptv>_singlefile.ptv and "-d 0 -v 3 -l 295"
        Then I expect the command to succeed
        And <folder>/output*.<fformat> is a single file
        When I check <folder>/output.<fformat> integrity with avprobe
        Then The video is OK
        When I launch videostitch-cmd with <folder>/<ptv>_multifiles.ptv and "-d 0 -v 3 -l 295"
        Then I expect the command to succeed
        And  The file size of <folder>/output*.<fformat> is below <limit> bytes
        When I check files <folder>/output*.<fformat> integrity with avprobe
        Then The videos are OK

        # TODO: reduce limit with VSA-7422
        Examples:
            | folder        | ptv                             | fformat | limit    |
            | videoformat01 | template_mp4_mpeg2_highbitrate  | mp4     | 15485760 |
            | videoformat01 | template_mp4_mpeg4_highbitrate  | mp4     | 15485760 |
            | videoformat01 | template_mov_mpeg2_highbitrate  | mov     | 15485760 |
            | videoformat01 | template_mov_mpeg4_highbitrate  | mov     | 15485760 |
            # The limit was too small on some machine for h264
            # On those ptvs the chunk size is 15 MB instead of 10MB
            | videoformat01 | template_mp4_h264_highbitrate   | mp4     | 27502928 |
            | videoformat01 | template_mov_h264_highbitrate   | mov     | 27502928 |

