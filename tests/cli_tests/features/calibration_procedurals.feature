# Testing calibration with procedurals

@slow
@calibration
Feature: calibration

    Scenario Outline: calibration with procedurals
      Given I use calibration_<ptv>.json for calibration
      When  I launch videostitch-cmd for calibration with calibration/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 3 "
      Then  I expect the command to succeed
      And   The JSON output calibration/scenes/<ptv>/output_calibration.ptv is valid

        Examples:
           | ptv                 |
           | procedurals         |
