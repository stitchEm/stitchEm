# A Test for comparing curve values in a sequence

@slow
Feature: curves applied on a sequence

    Scenario Outline: orientation on sequence
        When I launch videostitch-cmd with <project> and "-d 0 -f <frame> -l <frame>"
        Then I expect the command to succeed
        When I compare <output_file> with <reference_file><frame>.jpg
        Then I expect the comparison error to be less than <expected_error>

        Examples:
            | project                              | reference_file        | output_file                 | frame |  expected_error |
            # number-orientation.ptv: a single input. The video content has a visible number counting up with some orientation keyframes.
            # The visible numbers are have an offset over the frame number. The first video frame contains the number 4.
            # number.mp4 start_time: 0, first frame pts: 0
            | curveSequence/number_orientation.ptv | curveSequence/frame-  | curveSequence/curve-70.jpg  | 70    | 0.02           |
            | curveSequence/number_orientation.ptv | curveSequence/frame-  | curveSequence/curve-88.jpg  | 88    | 0.02           |
            | curveSequence/number_orientation.ptv | curveSequence/frame-  | curveSequence/curve-101.jpg | 101   | 0.02           |
            | curveSequence/number_orientation.ptv | curveSequence/frame-  | curveSequence/curve-120.jpg | 120   | 0.02           |
