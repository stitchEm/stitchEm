# Testing pano_coverage

@pano_coverage
Feature: pano_coverage

    Scenario Outline: pano_coverage
      Given I use pano_scoring.json for scoring
      When  I launch videostitch-cmd for scoring with pano_coverage/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 2 "
      Then  I expect the command to succeed
      When  I analyze uncovered_ratio of pano_coverage/scenes/<ptv>/output_scoring.ptv
      Then  I expect the full coverage to be <full_coverage>

        Examples:
           | ptv                | full_coverage |
           | factory            | true |
           | factory_with_holes | false |
