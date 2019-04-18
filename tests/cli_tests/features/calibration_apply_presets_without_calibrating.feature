# Test applying calibration presets without actually calibrating

@calibration
Feature: Applying Calibration Presets Without Calibrating

    Scenario Outline: apply calibration template
        Given I use calibration_apply_template_<ptv>.json for calibration presets application
        When  I launch videostitch-cmd for calibration presets application with calibration_apply_presets_without_calibrating/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 2 "
        Then  I expect the command to succeed
        And   I expect the geometries of calibration_apply_presets_without_calibrating/scenes/<ptv>/output_template.ptv are the same as calibration_apply_presets_without_calibrating/scenes/<ptv>/output_template_reference.ptv

        Examples:
            | ptv     |
            | factory |
