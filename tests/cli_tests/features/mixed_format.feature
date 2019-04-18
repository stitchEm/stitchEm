# Test to process ptv with different input formats (mixing static picture, procedural and video inputs)

@slow
Feature: Testing mixed (static/procedural/video) formats are working correctly together

    Scenario Outline: Mixed (static/procedural/video) formats
        When I launch videostitch-cmd with <folder>/<ptv> and "-d 0 -v 3 -f 0 -l 10"
        Then I expect the command to succeed
        When I check <folder>/output.mp4 integrity with avprobe
        Then The video is OK

        Examples:
            | folder        | ptv                                 |
            | videoformat01 | template_mp4_static_picture.ptv     |
            | videoformat01 | template_mp4_procedural.ptv         |
            | videoformat01 | template_mp4_static_procedural.ptv  |
