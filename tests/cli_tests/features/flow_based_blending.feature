# Test to see behaviour of Flow-based blending

Feature: Flow-based blending

    Scenario Outline: Flow-based blending
        When I launch videostitch-cmd with parallaxTolerantBlending/<test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare parallaxTolerantBlending/<test>-out-0.jpg with parallaxTolerantBlending/Reference-1-<test>-out-0.jpg
        Then I expect the comparison error to be less than 0.03

		Examples:
            | test         |
            | Rafting-1100 |
