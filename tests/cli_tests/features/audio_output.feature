# Test to process ptv with different audio output configuration

@slow
Feature: Testing different audio output configuration are working correctly

    Scenario Outline: Audio output formats
        When I test and check videostitch-cmd with "<audio_codec>" and "<sampling_rate>" and "<sample_format>" and "<channel_layout>" and "<audio_bitrate>"

        Examples:
            | audio_codec    | sampling_rate | sample_format | channel_layout               | audio_bitrate |
            | aac            | 44100 48000   | fltp          | mono stereo 3.0 4.0 5.0 5.1  | 192           |
            | mp3            | 44100 48000   | s16p          | mono stereo                  | 64 128 192    |
