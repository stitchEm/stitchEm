# Testing multi gpu

Feature: Seconary GPU

    Scenario Outline: Stitching with secondary GPU (if available)
        When I launch videostitch-cmd with multiGpu/<ptv> and "-d <n> -v 4 -f 0 -l 10"
        Then I expect the command to succeed

        Examples:
            | n | ptv                       |
            | 0 | template.ptv              |
            | 1 | template.ptv              |

    @slow
    Scenario Outline: Stitching with secondary GPU (if available)
        When I launch videostitch-cmd with multiGpu/<ptv> and "-d <n> -v 4 -f 0 -l 10"
        Then I expect the command to succeed

        Examples:
            | n | ptv                       |
            | 0 | template_procedural.ptv   |
            | 1 | template_procedural.ptv   |
            | 0 | template_procedural_2.ptv |
            | 1 | template_procedural_2.ptv |

