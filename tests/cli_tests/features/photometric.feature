Feature: Photometric calibration algorithm

    Scenario Outline: photometric
        Given I use photometric_calibration.json for photometric calibration
        When  I launch videostitch-cmd for photometric calibration with photometric/factory/factory.ptv and "-f 0 -l 0 "
        Then  I expect the command to succeed
        And   The photometric output photometric/factory/output_photo_calib.ptv is valid
        And   The exposure RGB score in photometric/factory/expo_score.ptv is less than <rdiff_pre>, <gdiff_pre>, <bdiff_pre>
        And   The exposure RGB score in photometric/factory/expo_score_post_photo_calib.ptv is less than <rdiff_post>, <gdiff_post>, <bdiff_post>

        Examples:
        | rdiff_pre | gdiff_pre | bdiff_pre | rdiff_post | gdiff_post | bdiff_post |
        | 28        | 27        | 22        | 29         | 28         | 23         |
