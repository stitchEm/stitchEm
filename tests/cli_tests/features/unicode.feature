# This test is messing around with unicode characters

Feature: unicode

    Scenario: unicode video to ascii picture
        When I launch videostitch-cmd with unicode/load_unicode_video___write_ascii_image.ptv and "-d 0 -f 18 -l 18"
        Then I expect the command to succeed
        When I compare unicode/vs-out-18.jpg with unicode/reference.jpg
        Then I expect the comparison error to be less than 0.03

    Scenario: unicode video to unicode picture
        When I launch videostitch-cmd with unicode/load_unicode_video___write_unicode_image.ptv and "-d 0 -f 18 -l 18"
        Then I expect the command to succeed
        When I rename unicode/hebrew_אָלֶף־בֵּית עִבְ_dummy_image-18.jpg to unicode/unicode_output_image.jpg
        And  I compare unicode/unicode_output_image.jpg with unicode/reference.jpg
        Then I expect the comparison error to be less than 0.03

    Scenario: ascii video to ascii video
        When I launch videostitch-cmd with unicode/load_ascii_video___write_ascii_video.ptv and "-d 0 -f 0 -l 18"
        Then I expect the command to succeed
        When I check unicode/ascii_output_video.mp4 integrity with avprobe
        Then The video is OK

    Scenario: unicode video to ascii video
        When I launch videostitch-cmd with unicode/load_unicode_video___write_ascii_video.ptv and "-d 0 -f 0 -l 18"
        Then I expect the command to succeed
        When I check unicode/output_video.mp4 integrity with avprobe
        Then The video is OK

    Scenario: unicode video to unicode video
        When I launch videostitch-cmd with unicode/load_unicode_video___write_unicode_video.ptv and "-d 0 -f 0 -l 18"
        Then I expect the command to succeed
        When I rename unicode/hebrew_אָלֶף־בֵּית עִבְ_output_video.mp4 to unicode/output_video.mp4
        And  I check unicode/output_video.mp4 integrity with avprobe
        Then The video is OK
