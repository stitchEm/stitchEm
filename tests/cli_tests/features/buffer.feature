# Test to process ptv with different buffer size

@slow
Feature: different buffer size

    Scenario Outline: Buffer size
        When I launch videostitch-cmd with buffer0<n>/template<buffer_size>.ptv and "-d 0 -v 3 -f 0 -l 10"
        Then I expect the command to succeed

        Examples:
            | n | buffer_size |
            | 1 | 0           |
            | 1 | 1           |
            | 2 | 0           |
            | 2 | 1           |
            | 3 | 0           |
            | 3 | 1           |

