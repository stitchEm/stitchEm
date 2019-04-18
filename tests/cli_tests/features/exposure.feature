# Test algo expo_params

Feature: exposure

    Scenario: expo_params
        When I launch videostitch-cmd with exposure01/template.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed

