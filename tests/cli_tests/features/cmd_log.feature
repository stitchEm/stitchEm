# Test to see behaviour of log in command line

Feature: CMD Log

    Scenario Outline: CMD Log
        When I launch videostitch-cmd with laplacianBlending/<test>.ptv and "-d 0 -f 0 -l 0 -v 3"
        Then I expect the command to succeed and stdout contains "stitched frame at "

		Examples:
            | test        |
            | Rafting     |