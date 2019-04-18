# Testing mask

Feature: mask

    Scenario Outline: mask
      Given I use mask-seam.json for mask
      When I launch videostitch-cmd for mask with blendingMask/<ptv>.ptv and " -d 0 -v 0 "
      Then I expect the command to succeed
      When I launch videostitch-cmd with generated blendingMask/outputFinal.ptv and "-d 0 -v 0 -f 0 -l 0 "
      Then I expect the command to succeed
      When I compare blendingMask/out-vs-0.jpg with blendingMask/Reference-<ptv>-distortion-size-order-seam.jpg
      Then I expect the comparison error to be less than 0.03

        Examples:
            | ptv        |
            | Rafting    |
