# Processing autocrop 

@autocrop

Feature: autocrop 
    Scenario Outline: autocrop       
        Given I use autoCrop.json for autocrop
        When I launch videostitch-cmd for autocrop with autoCrop/Ptv/<test>.ptv and " -d 0 -v 0 "
        Then I expect the command to succeed
        When I compare autoCrop/Ptv/<test>/input-00.jpg_circle.png with autoCrop/Ptv/<test>/reference-input-00.jpg_circle.png
        Then I expect the comparison error to be less than 0.03    	        
		
        Examples:
            | test       |
            | Fountain   |