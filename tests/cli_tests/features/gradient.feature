# Processing ptv with different gradient values

@slow
Feature: different gradient values

    Scenario Outline: different gradient values for image inputs
        When I launch videostitch-cmd with gradient/<test>.ptv and "-d 0 -v 3 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare gradient/<test>-out-0.jpg with gradient/Reference<test>-out-1.jpg
        Then I expect the comparison error to be less than 0.03

        Examples:
            | test           |
            | template_1     |
            | template_2     |
            | template_3     |
            | template_4     |
            | DestinoWide    |
            | Fountain       |
            | RaceBoat       |
            | Rafting        |
            | Scuba          |

    Scenario Outline: different gradient values for procedural inputs
        When I launch videostitch-cmd with gradient/<test>.ptv and "-d 0 -v 3 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare gradient/<test>-out-1.jpg with gradient/Reference<test>-out-1.jpg
        Then I expect the comparison error to be less than 0.03

        Examples:
            | test           |
            | RES-512-zenith |
            | 6_rect_inputs  |
