# A non regression test for the exposure correction

Feature: exposure correction

    Scenario Outline: exposure correction
        When I launch videostitch-cmd with exposureCorrection/<exposure_setting>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare exposureCorrection/<exposure_setting>-1.png with exposureCorrection/<exposure_setting>-1-reference.png
        Then I expect the comparison error to be less than 0.03

        Examples:
            | exposure_setting                   |
            | checker_6_emor_exposure            | # Exposure correction on inputs with EMoR photo correction
            | checker_6_emor_exposure_color_corr | # Color (rgb) correction on inputs with EMoR photo correction

    @slow
    Scenario Outline: exposure correction
        When I launch videostitch-cmd with exposureCorrection/<exposure_setting>.ptv and "-d 0 -f 0 -l 0"
        Then I expect the command to succeed
        When I compare exposureCorrection/<exposure_setting>-1.png with exposureCorrection/<exposure_setting>-1-reference.png
        Then I expect the comparison error to be less than 0.03

        Examples:
            | exposure_setting                   |
            | checker_6_exposure_everything      | # Combination of settings from all other sample projects
            | checker_6_gamma_color_corr         | # Color (rgb) correction on inputs with Gamma photo correction
            | checker_6_gamma_exposure           | # Exposure correction on inputs with Gamma photo correction
            | checker_6_global_ev_color_corr     | # Output / project ev and rgb correction settings
            | checker_6_linear_color_corr        | # Exposure correction on inputs with linear photo correction
            | checker_6_linear_exposure          | # Color (rgb) correction on inputs with EMoR photo correction
            | checker_6_no_correction            | # Checkerboard without any exposure or color corrections
            | checker_6_vignette                 | # Vignette correction on inputs
