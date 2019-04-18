@slow
Feature: Regression tests for the debug mergers

    Scenario Outline: Different mergers
        When I launch videostitch-cmd with mergers/<test>.ptv and "-d 0 -v 3 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare mergers/<test>-out-<frame>.<ext> with mergers/<reference>-out-<frame>.<ext>
        Then I expect the comparison error to be less than 0.008

        Examples:
            | test                               | frame | reference                                      | ext |
            | 6_rect_inputs_checker              | 1     | Reference-1-6_rect_inputs_checker              | png |
            | paris_office_terrace/array         | 0     | paris_office_terrace/Reference-array           | png |
            | paris_office_terrace/diff          | 0     | paris_office_terrace/Reference-1-diff          | jpg |
            | paris_office_terrace/exposure_diff | 0     | paris_office_terrace/Reference-1-exposure_diff | jpg |
            | paris_office_terrace/noblend       | 0     | paris_office_terrace/Reference-1-noblend       | jpg |
            | paris_office_terrace/stack         | 0     | paris_office_terrace/Reference-1-stack         | jpg |
