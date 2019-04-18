# Test motion synchro

@slow
@motionsynchro
Feature: Synchronization

    Scenario Outline: motion synchro
        Given I use motion_sync_<tst>.json for synchronization
        When  I launch videostitch-cmd for synchronization with motion_synchro/<tst>/<tst>.ptv and "-f 0 -l 10 -v 4 "
        Then  I expect the command to succeed
        And   The synchronization output "motion_synchro/<tst>/motionout.ptv" is valid
        And   The synchronization output "motion_synchro/<tst>/motionout.ptv" is consistent with "motion_sync_<tst>_ref.ptv"

        Examples:
            | tst    |
            | small  |
            | medium |
            | large  |

