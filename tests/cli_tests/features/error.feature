# Test to see behaviour of cmd-line with wrong input images

Feature: Invalid inputs

    Scenario Outline: Invalid pictures
        When I launch videostitch-cmd with imageError/template_<n>.ptv and "-d 0 -v 3 -f 0 -l 10"
        Then I expect the command to fail with code 1

        Examples:
            | n |
            | 1 |
            | 2 |
            | 3 |
            | 4 |

    Scenario Outline: Invalid seek range
        When I launch videostitch-cmd with videoError/<ptv> and "-d 0 -v 3 -f 0 -l 10"
        Then I expect the command to succeed

        Examples:
            | ptv                             |
            | template_seek_range_error.ptv   |
            | template_seek_range_error_2.ptv |
            | template_seek_range_error_3.ptv |
            | template_seek_range_error_4.ptv |

