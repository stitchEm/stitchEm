# Testing calibration for Orah 4i

@slow
@calibration
Feature: calibration for Orah 4i

    Scenario Outline: calibration for Orah 4i
      Given I use calibration_<ptv>.json for calibration
      When  I launch videostitch-cmd for calibration with calibration_orah_4i/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 3 "
      Then  I expect the command to succeed
      When  I analyze score of calibration_orah_4i/scenes/<ptv>/output_scoring.ptv
      Then  I expect the score to be more than 0.75
      When  I analyze uncovered_ratio of calibration_orah_4i/scenes/<ptv>/output_scoring.ptv
      Then  I expect the full coverage to be <full_coverage>
      And   The calibration cost of output "calibration_orah_4i/scenes/<ptv>/output_calibration.ptv" is consistent with "calibration_<ptv>_ref.ptv"
      When  I compare calibration_orah_4i/scenes/<ptv>/<ptv>-vs-out-0.png with calibration_orah_4i/scenes/<ptv>/reference-1-vs-out-0.png
      Then  I expect the comparison error to be less than 0.005

        Examples:
           | ptv                               | full_coverage |
           | AQ1610012338                      | true |
           | AQ1610012338_artificial_keypoints | true |
