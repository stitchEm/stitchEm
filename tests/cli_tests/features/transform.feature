# A non regression test for the transform stack

Feature: transformstack

    Scenario Outline: mtransformstack
        When I launch videostitch-cmd with transformstack/<ptv>/<ptv>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare transformstack/<ptv>/<ptv>_output-0.jpg with transformstack/<ptv>/reference-1.jpg
        Then I expect the comparison error to be less than 0.03

        Examples:
            | ptv      |
            | circular |
            | elmo     |

    @slow
    Scenario Outline: projections
        When I launch videostitch-cmd with transformstack/regression/<in>_<out>.ptv and "-d 0 -l 0"
        Then I expect the command to succeed
        When I compare transformstack/regression/<in>_<out>-1.png with transformstack/regression/<in>_<out>-2-reference.png
        Then I expect the comparison error to be less than 0.03

        Examples:
            | in     | out    |
            | cf     | cf     |
            | cf     | erect  |
            | cf     | ff     |
            | cf     | rect   |
            | cf     | stereo |
            | erect  | cf     |
            | erect  | erect  |
            | erect  | ff     |
            | erect  | rect   |
            | erect  | stereo |
            | ff     | cf     |
            | ff     | erect  |
            | ff     | ff     |
            | ff     | rect   |
            | ff     | stereo |
            | rect   | cf     |
            | rect   | erect  |
            | rect   | ff     |
            | rect   | rect   |
            | rect   | stereo |

    @slow
    Scenario Outline: pano parameters
        When I launch videostitch-cmd with transformstack/regression/<in>_<out>_<param>_<value1>_<value2>.ptv and "-d 0 -l 0"
        Then I expect the command to succeed
        When I compare transformstack/regression/<in>_<out>_<param>_<value1>_<value2>-1.png with transformstack/regression/<in>_<out>_<param>_<value1>_<value2>-2-reference.png
        Then I expect the comparison error to be less than 0.03

        Examples:
            | in | out    | param | value1           | value2 |
            | ff | erect  | size  | 64               | 128-v2 |
            | ff | erect  | size  | 256              | 512    |
            | ff | erect  | size  | 1024             | 2048   |
            | ff | erect  | size  | 2048             | 4096   |
            | ff | stereo | size  | 128              | 1024   |
            | ff | stereo | size  | 1024             | 1024   |
            | ff | erect  | hfov  | 2                | 0-v2   |
            | ff | erect  | hfov  | 8                | 0-v2   |
            | ff | erect  | hfov  | 45               | 0-v2   |
            | ff | erect  | hfov  | 90               | 0      |
            | ff | erect  | hfov  | 150              | 0      |
            | ff | erect  | hfov  | 220              | 0      |
            | ff | erect  | hfov  | 360              | 0      |
            | ff | stereo | hfov  | 2                | 0-v2   |
            | ff | stereo | hfov  | 8                | 0-v2   |
            | ff | stereo | hfov  | 45               | 0-v2   |
            | ff | stereo | hfov  | 90               | 0      |
            | ff | stereo | hfov  | 150              | 0      |
            | ff | stereo | hfov  | 220              | 0      |
            | ff | stereo | hfov  | 350              | 0      |
            | ff | erect  | crop  | l500             | 0      |
            | ff | erect  | pad   | t500             | 0      |
            | ff | erect  | pad   | b500             | 0      |
            | ff | erect  | pad   | t100b100         | 0      |
            | ff | erect  | yaw   | 45               | 0      |
            | ff | erect  | yaw   | 90               | 0      |
            | ff | erect  | yaw   | 270              | 0      |
            | ff | erect  | pitch | 45               | 0      |
            | ff | erect  | pitch | 90               | 0      |
            | ff | erect  | pitch | 270              | 0      |
            | ff | erect  | roll  | 45               | 0      |
            | ff | erect  | roll  | 90               | 0      |
            | ff | erect  | roll  | 270              | 0      |
#           | ff | erect  | crop  | r500             | 0      |
#           | ff | erect  | crop  | b500             | 0      |
#           | ff | erect  | crop  | t500             | 0      |
#           | ff | erect  | crop  | l200r200t200b200 | 0      |

    Scenario Outline: input parameters
        When I launch videostitch-cmd with transformstack/inputs/test_input_<setting>.ptv and "-d 0 -l 0"
        Then I expect the command to succeed
        When I compare transformstack/inputs/input_<setting>-1.png with transformstack/inputs/input_<setting>-1-reference-1.png
        Then I expect the comparison error to be less than 0.03

        Examples:
            | setting      |
            # VSA-4791
            #| crop         |
            | dist         |
            | hfov         |
            | roll         |
            | width_height |

    Scenario Outline: projections board
        When I launch videostitch-cmd with transformstack/regression/<in>_<out>_board.ptv and "-d 0 -l 0"
        Then I expect the command to succeed
        When I compare transformstack/regression/<in>_<out>_board-1.png with transformstack/regression/<in>_<out>_board-1-reference-1.png
        Then I expect the comparison error to be less than 0.0004

        Examples:
            | in     | out    |
            | erect  | erect  |

    Scenario Outline: Stitching identity accuracy
        When I launch videostitch-cmd with input_formats/<test>-<format>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare input_formats/<format>-out-0.<format> with input_formats/input.<format>
        Then I expect the comparison error to be less than <expected_error>

		Examples:
            | test               | format | expected_error |
            | identity           | jpg    | 0.014          |
            | identity           | png    | 0.013          |
            | flip-unflip        | png    | 0.015          |
