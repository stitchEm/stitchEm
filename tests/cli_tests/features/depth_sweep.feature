Feature: Depth map computation

    Scenario: Pyramid sweep
        When I launch depth with depth/office-nov/frame_29900.ptv and "-f 0 -l 0 -s 4"
        Then I expect the command to succeed
        When I compare depth/office-nov/office-nov-depth-depth-0-0.png with depth/office-nov/office-nov-depth-depth-0-0-ref-171204.png
        Then I expect the comparison error to be less than 0.0013
