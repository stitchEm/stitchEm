# A non regression test for frame seeking with libavReader

@seek
Feature: seek frame

    Scenario Outline: seek frame
        When I launch videostitch-cmd with <project> and "-d 0 -f <frame> -l <frame>"
        Then I expect the command to succeed
        When I compare <output_file><frame>.jpg with <reference_file>-<frame>.jpg
        Then I expect the comparison error to be less than <expected_error>

        Examples:
            | project                          | reference_file                           | output_file                     | frame | expected_error |
            # number.ptv: a single input. The video content has a visible number counting up.
            # The visible numbers are have an offset over the frame number. The first video frame contains the number 4.
            # number.mp4 start_time: 0, first frame pts: 0
            | seekFrame/number.ptv              | seekFrame/frame                         | seekFrame/seekFrameOut-         | 396   | 0.02           |
            | seekFrame/number.ptv              | seekFrame/frame                         | seekFrame/seekFrameOut-         | 496   | 0.02           |
            | seekFrame/number.ptv              | seekFrame/frame                         | seekFrame/seekFrameOut-         | 153   | 0.02           |
            | seekFrame/number.ptv              | seekFrame/frame                         | seekFrame/seekFrameOut-         | 986   | 0.02           |
            | seekFrame/number.ptv              | seekFrame/frame                         | seekFrame/seekFrameOut-         | 1000  | 0.02           |

    @slow
    Scenario Outline: seek frame
        When I launch videostitch-cmd with <project> and "-d 0 -f <frame> -l <frame>"
        Then I expect the command to succeed
        When I compare <output_file>-<frame>.jpg with <reference_file>-<frame>.jpg
        Then I expect the comparison error to be less than <expected_error>

        Examples:
            | project                           | reference_file                          | output_file                    | frame | expected_error |
            # delayed.ptv contains 6 inputs of 3 seconds length with synthetic video offset of 1 second
            # delayed video start_time: 9000, first frame pts: 9000
            | delayed_video_stream/delayed.ptv  | delayed_video_stream/delayed-reference1 | delayed_video_stream/delayed   | 0     | 0.02           |
            | delayed_video_stream/delayed.ptv  | delayed_video_stream/delayed-reference1 | delayed_video_stream/delayed   | 24    | 0.02           |
            | delayed_video_stream/delayed.ptv  | delayed_video_stream/delayed-reference  | delayed_video_stream/delayed   | 49    | 0.02           |
            # videoformat01 video start_time: 6006, first frame pts: 0
            | videoformat01/seekFrame.ptv       | videoformat01/seekFrame-reference       | videoformat01/seekFrame        | 0     | 0.02           |
            | videoformat01/seekFrame.ptv       | videoformat01/seekFrame-reference       | videoformat01/seekFrame        | 300   | 0.02           |
            # computed difference is a little higher on mac_opencl, but frame seems visually to be the correct one
            | videoformat01/seekFrame.ptv       | videoformat01/seekFrame-reference       | videoformat01/seekFrame        | 2957  | 0.025          |
            # test seek frame with mpeg4 input, same input with different offsets
            | videoformat01/seekFrame_mpeg4.ptv | videoformat01/seekFrame-mpeg4-reference1| videoformat01/seekFrame-mpeg4  | 1337  | 0.02           |
            | seekFrame/GoPro.ptv               | seekFrame/GoPro-reference               | seekFrame/GoPro                | 1000  | 0.02           |
            # PTS of frames > 1500 are not exactly the target PTS
            # Should still seek to the correct frame
            | seekFrame/GoPro.ptv               | seekFrame/GoPro-reference               | seekFrame/GoPro                | 2000  | 0.02           |
