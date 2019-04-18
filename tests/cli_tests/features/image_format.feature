# Test to verify still images output from cmd-line are correct

Feature: Command line output

    Scenario: Command line
        When I launch videostitch-cmd with imageFormat/template.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare imageFormat/output-0.jpg with imageFormat/hugin.jpg
        Then I expect the comparison error to be less than 0.03
