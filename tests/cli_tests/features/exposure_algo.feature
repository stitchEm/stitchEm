# Test exposure algorithm

Feature: exposure Algo

    Scenario Outline: exposure Algo
        Given I use exposure_<n>_<stabilize>.json for exposure
        When  I launch videostitch-cmd for exposure with videoformat01/full_1024.ptv and "-f 0 -l 10 "
        Then  I expect the command to succeed
        And   The exposure output ptv is valid
        And   The exposure RGB score in videoformat01/output_exposure_scoring.ptv is less than <rdiff>, <gdiff>, <bdiff>

        Examples:
            | n | stabilize | rdiff | gdiff | bdiff |
            | 0 | True      | 14    | 14    | 18    |
            | 0 | False     | 14    | 14    | 18    |

    @slow
    Scenario Outline: exposure Algo
        Given I use exposure_<n>_<stabilize>.json for exposure
        When  I launch videostitch-cmd for exposure with videoformat01/full_1024.ptv and "-f 0 -l 10 "
        Then  I expect the command to succeed
        And   The exposure output ptv is valid
        And   The exposure RGB score in videoformat01/output_exposure_scoring.ptv is less than <rdiff>, <gdiff>, <bdiff>

        Examples:
            | n | stabilize | rdiff | gdiff | bdiff |
            | 1 | True      | 15    | 16    | 21    |
            | 1 | False     | 17    | 17    | 24    |
            | 2 | True      | 16    | 16    | 23    |
            | 2 | False     | 17    | 17    | 24    |
            | 3 | True      | 12    | 13    | 13    |
            | 3 | False     | 14    | 14    | 15    |
            | 4 | True      | 10    | 11    | 13    |
            | 4 | False     | 10    | 11    | 14    |
            | 5 | True      | 14    | 14    | 15    |
            | 5 | False     | 14    | 14    | 18    |
