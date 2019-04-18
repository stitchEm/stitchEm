# Test to see behaviour of Laplacian blending

Feature: Laplacian blending

    Scenario Outline: Laplacian blending
        When I launch videostitch-cmd with laplacianBlending/<test>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare laplacianBlending/<test>-out-0.jpg with laplacianBlending/Reference<test>-out-0.jpg
        Then I expect the comparison error to be less than 0.03

		Examples:
            | test        |
            | DestinoWide |
            | Fountain    |
            | RaceBoat    |
            | Rafting     |
            | Scuba       |
