# Test to process ptv with a typical audio pipe configuration

Feature: Testing a ptv project with a typical audio pipe configuration


    Scenario Outline: Audio process
        When I launch videostitch-cmd with videoformat01/<ptv> and "-f 0 -l 100"
        Then The video is OK

        Examples:
            | ptv                 |
            | template_audio.ptv  |

