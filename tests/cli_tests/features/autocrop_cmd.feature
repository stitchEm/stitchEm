# Processing autocrop comand-line

Feature: autocrop cmd

    Scenario Outline: autocrop command-line
        When I launch autocrop-cmd with input autoCrop/Doc/<test>.png and output output-<test>.json
        Then I expect the command to succeed
        When I compare autoCrop/Doc/<test>.png_circle.png with autoCrop/Doc/<test>.png_circle_reference.png
        Then I expect the comparison error to be less than 0.02        
		
        Examples:
            | test       |
            | wedding-07 |