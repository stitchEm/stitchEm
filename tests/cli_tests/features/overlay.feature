# Testing overlay

Feature: overlay

    Scenario Outline: overlay
        When I launch videostitch-cmd with overlay/<ptvfile>.ptv and "-d 0 -f 0 -l <lastFrame>"
        Then I expect the command to succeed
        When I compare overlay/<ptvfile>-out-<lastFrame>.png with overlay/<ptvfile>-rf.png
        Then I expect the comparison error to be less than 0.02

        Examples:
           | ptvfile                           | lastFrame    |
           | overlay-1-logo                    |  0           |
           | overlay-1-logo-alpha              |  5           |
           | overlay-1-logo-6dof               |  5           |
           | overlay-1-logo-global-orientation |  5           |
           | overlay-2-logos                   |  5           |
