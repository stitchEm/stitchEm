# Testing audio synchronization

@slow
Feature: Synchronization

    Scenario Outline: audio synchro
        Given I use audio_sync_<tst>.json for synchronization
        When  I launch videostitch-cmd for synchronization with audio_synchro/<tst>/<tst>.ptv
        Then  I expect the command to succeed
        And   The synchronization output "audio_synchro/<tst>/audio_sync_out.ptv" is valid
        And   The synchronization output "audio_synchro/<tst>/audio_sync_out.ptv" is consistent with "audio_sync_<tst>_ref.ptv" within 0 frames

        Examples:
            | tst     |
            | grand_8 |

    Scenario Outline: audio synchro on corrupted inputs
        Given I use audio_sync_moto.json for synchronization
        When  I launch videostitch-cmd for synchronization with audio_synchro/<tst>/<tst>.ptv
        Then  I expect the command to succeed

        Examples:
            | tst     |
            | moto    |
