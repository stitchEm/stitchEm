# Test Calibration Deshuffling

@slow
@calibration
Feature: Calibration Deshuffling

    Scenario Outline: calibration deshuffling
        Given I use <config>.json for calibration deshuffling
        When  I launch videostitch-cmd for calibration deshuffling with calibration_deshuffling/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 4 "
        Then  I expect the command to succeed
        # The input PTV files were modified to have the "stack_order" set to the input number, to check the reshuffling order of input definitions
        And   I expect the input readers and stack orders of calibration_deshuffling/scenes/<ptv>/output_calibration.ptv are the same as calibration_deshuffling/scenes/<ptv>/<config>_ref.ptv

        Examples:
            | ptv                             | config                                     |
            | CageFlight-frame-0-uncalibrated | H3PRO6-1440p-deshufle                      |
            | CageFlight-frame-0-shuffled     | H3PRO6-1440p-deshufle                      |
            | CageFlight-frame-0-shuffled     | H3PRO6-1440p-deshufle-only                 |
            | CageFlight-frame-0-shuffled     | H3PRO6-1440p-deshufle-only-preserve-inputs |
            # | CageFlight-frame-500-calibrated | H3PRO6-1440p-deshufle                      |
            # | CageFlight-frame-500-calibrated | H3PRO6-1440p-deshufle-only                 |
            # | CageFlight-frame-500-shuffled   | H3PRO6-1440p-deshufle-only                 |
            # | CageFlight-frame-500-shuffled   | H3PRO6-1440p-deshufle-only-preserve-inputs |
