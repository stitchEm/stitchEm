# Test calibration presets maker

@calibration
Feature: Calibration Presets Maker

    Scenario Outline: calibration presets maker
        Given I use calibration_presets_maker_<ptv>.json for calibration presets maker
        When  I launch videostitch-cmd for calibration presets maker with calibration_presets_maker/scenes/<ptv>/<ptv>.ptv and " -d 0 -v 2 "
        Then  I expect the command to succeed
        And   I expect calibration_presets_maker/scenes/<ptv>/output_presets.json is the same as calibration_presets_maker/scenes/<ptv>/output_presets_oriented_translations_ref.json with 6 digits after the decimal point for float
        And   I expect calibration_presets_maker/scenes/<ptv>/output_final_presets.ptv is the same as calibration_presets_maker/scenes/<ptv>/output_final_presets_oriented_translations_ref.ptv with 6 digits after the decimal point for float

        Examples:
            | ptv     |
            | factory |
