Feature: Test perf

    @perf @prof
    Scenario Outline: videoformat01
        When I launch videostitch-cmd with videoformat01/<ptv>.ptv
        Then I expect it to take less than <timeout> seconds
        When I profile videostitch-cmd with videoformat01/<ptv>.ptv
        Then I expect it to take less than <GPUtime> seconds

        Examples:
            | ptv                                      | timeout | GPUtime |
            | full_2048                                | 600     | 540     |
            | full_2048_coreV2                         | 240     | 220     |
            | template_mp4_h264_highbitrate_singlefile | 80      | 80      |

    @perf
    Scenario: template seek
        When I launch videostitch-cmd with videoformat01/template_seek.ptv
        Then I expect it to take less than 1600 seconds

    @perf @prof
    Scenario Outline: transform stack
        When I launch videostitch-cmd with procedural transformstack/perf/<in>_<out>.ptv
        Then I expect it to take less than <timeout> seconds
        When I profile videostitch-cmd with procedural transformstack/perf/<in>_<out>.ptv
        Then I expect it to take less than <GPUtime> seconds

        Examples:
            | in     | out    | timeout | GPUtime |
            | cf     | cf     | 5.1     | 3.8     |
            | cf     | erect  | 6.1     | 8.8     |
            | cf     | ff     | 6.1     | 9.0     |
            | cf     | rect   | 6.1     | 7.5     |
            | cf     | stereo | 5.1     | 4.3     |
            | erect  | cf     | 4.1     | 3.6     |
            | erect  | erect  | 6.1     | 8.6     |
            | erect  | ff     | 7.1     | 8.8     |
            | erect  | rect   | 6.1     | 6.9     |
            | erect  | stereo | 6.1     | 4.3     |
            | ff     | cf     | 4.1     | 3.8     |
            | ff     | erect  | 6.1     | 8.7     |
            | ff     | ff     | 6.1     | 9.0     |
            | ff     | rect   | 6.1     | 7.8     |
            | ff     | stereo | 6.1     | 5.2     |
            | rect   | cf     | 3.1     | 3.6     |
            | rect   | erect  | 4.1     | 6.1     |
            | rect   | ff     | 4.1     | 4.6     |
            | rect   | rect   | 4.1     | 6.1     |
            | rect   | stereo | 4.1     | 6.1     |

 

