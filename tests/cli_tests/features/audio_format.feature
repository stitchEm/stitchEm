# Test to process ptv with different video input formats

@slow
Feature: Testing different audio formats are working correctly

    Scenario Outline: Video formats
        When I launch videostitch-cmd with videoformat01/<ptv> and "-f 0 -l 200"
        Then I expect the command to succeed
        Then I check the audio bitrate of videoformat01/output.<extension> to be equal to <bitrate>

        Examples:
            | bitrate | ptv                         | extension |
            | 64      | template_aac_64kbs.ptv      | mp4       |
            | 96      | template_aac_96kbs.ptv      | mp4       |
            | 128     | template_aac_128kbs.ptv     | mp4       |
            | 192     | template_aac_192kbs.ptv     | mp4       |
            | 64      | template_mp3_64kbs.ptv      | mp4       |
            | 96      | template_mp3_96kbs.ptv      | mp4       |
            | 128     | template_mp3_128kbs.ptv     | mp4       |
            | 192     | template_mp3_192kbs.ptv     | mp4       |
            | 192     | template_pcm_aac_192kbs.ptv | mov       |
