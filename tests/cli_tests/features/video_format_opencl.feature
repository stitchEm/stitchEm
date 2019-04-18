# Test to process ptv with different video input formats

@slow
Feature: Testing different video formats are working correctly

    Scenario Outline: Video formats
        When I launch videostitch-cmd with <folder>/<ptv> and "-d 0 -v 3 -f <first> <last>"
        Then I expect the command to succeed
        When I check <folder>/output.<extension> integrity with avprobe
        Then The video is OK

        Examples:
            | folder        | ptv                        | extension | first | last  |
            | videoformat01 | template_linear_mp4.ptv    | mp4       | 0     | -l 10 |
            | videoformat01 | template_mp4_prores.ptv    | mov       | 0     | -l 10 |
            | videoformat01 | template_mpeg4_mp4.ptv     | mp4       | 0     | -l 10 |
            | videoformat01 | template_prores_prores.ptv | mov       | 0     | -l 10 |
            | videoformat01 | template_prores_prores.ptv | mov       | 1470  |       |
            | videoformat01 | template_prores_mp4.ptv    | mp4       | 0     | -l 10 |
