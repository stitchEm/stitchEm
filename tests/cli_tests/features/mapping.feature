# mappingPtv

@wip
Feature: mappingPtv

    Scenario Outline: mappingPtv
        When I launch videostitch-cmd with <folder>/template.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare <folder>/output-1.jpg with <folder>/ref.jpg
        Then I expect the comparison error to be less than 0.03

        Examples:
            | folder       |
            | mappingPtv01 |
            | mappingPtv02 |

