# A really simple test to verify that a project without audio track in the videos
# does not crash

@slow
Feature: no audio track

    Scenario Outline: No audio track
        When I launch videostitch-cmd with no_audio/no_audio.ptv and "-v 4"
        Then I expect the command to succeed

        Examples:
            | args                    |
            | -v 4                    |


