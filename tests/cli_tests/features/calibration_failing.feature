# Testing calibration

@slow
@calibration
Feature: calibration

    Scenario Outline: calibration failure
      Given I use calibration_<json>.json for calibration
      When  I launch videostitch-cmd for calibration with calibration/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 2 "
      Then  I expect the command to fail and stderr contains "Could not apply algo 'calibration'"

        Examples:
           | ptv                               | json    |
           | louvre                            | factory |
           | factory                           | louvre  |
           | factory_incremental               | louvre  |
           | AQ1610012338_artificial_keypoints | AQ1610012338_artificial_keypoints |
