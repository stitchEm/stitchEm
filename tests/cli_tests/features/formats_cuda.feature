# Test to process ptv with different video input/output formats

Feature: Testing different video formats are working correctly

    Scenario Outline: Video formats
        When I launch videostitch-cmd with formats/<ptv>.ptv and "-d 0 -f 1 -l 200"
        Then I expect the command to succeed
        When I compare video formats/output.<extension> to formats/<ptv>_v2.<extension>
        Then mse_avg is under 0.10
        And  I expect the number of frames of formats/output.<extension> to be 200
        When I check formats/output.<extension> integrity with avprobe
        Then The video is OK

        Examples:
            | ptv        | extension |
            | mov_h264   | mov       |
            # VSA-7077
            #| mov_mpeg2  | mov       |
            | mov_mjpeg  | mov       |

    @slow
    Scenario Outline: Video formats
        When I launch videostitch-cmd with formats/<ptv>.ptv and "-d 0 -f 1 -l 200"
        Then I expect the command to succeed
        When I compare video formats/output.<extension> to formats/<ptv>_v2.<extension>
        Then mse_avg is under 0.10
        And  I expect the number of frames of formats/output.<extension> to be 200
        When I check formats/output.<extension> integrity with avprobe
        Then The video is OK

        Examples:
            | ptv        | extension |
            | mov_prores | mov       |
            | mp4_h264   | mp4       |
            | mp4_mpeg2  | mp4       |
            | mp4_mjpeg  | mp4       |

