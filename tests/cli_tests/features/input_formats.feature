# Test different input formats

Feature: Input formats

    Scenario Outline: Input image formats
        When I launch videostitch-cmd with input_formats/<format>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare input_formats/<format>-out-0.jpg with input_formats/<format>-rf.jpg
        Then I expect the comparison error to be less than 0.02

		Examples:
            | format |
            | jpg    |
            | png    |

    @wip
    Scenario Outline: Video input
        When I launch videostitch-cmd with input_formats/<format>.ptv and "-d 0 -f 0 -l 10"
        # VSA-5605: studio displays an error for those videos
        Then I expect the command to fail with code 1
        # Let's fix VSA-6354 first
        #Then I expect the command to suceed
        #When I compare input_formats/<format>.jpg with input_formats/<format>-rf.jpg
        #Then I expect the comparison error to be less than 0.02

    Examples:
            | format |
            | mpg    |
            | mkv    |
            | mp2    |
            | mpeg   |
            | ogv    |
            | ogg    |
            | wmv    |

    Scenario: Garmin Virb camera
        When I launch videostitch-cmd with input_formats/garmin.ptv and "-d 0 -f 1 -l 10"
        Then I expect the command to succeed
        When I compare input_formats/garmin-out-1.jpg with input_formats/garmin-ref-1.jpg
        Then I expect the comparison error to be less than 0.01

