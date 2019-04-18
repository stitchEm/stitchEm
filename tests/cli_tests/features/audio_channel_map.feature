# Test to the output audio channel map for files and rtmp streams

Feature: Testing different audio output configuration are working correctly

    Scenario: Check channel map in a recorded file
        When  I launch videostitch-cmd with audio_channel_map/sin-gen-stereo.ptv and "-d 0 -f 0 -l 33"
        Then  I expect the output audio channel map of audio_channel_map/output.mp4 to be "0 1"
        When  I launch videostitch-cmd with audio_channel_map/sin-gen-amb-wxyz.ptv and "-d 0 -f 0 -l 33"
        # Internally we have a WXYZ channel order but by default we record an AMBIX format (WYZX) TBC
        Then  I expect the output audio channel map of audio_channel_map/output.mp4 to be "0 2 3 1"

    @slow
    Scenario Outline: Check channel map in a RTMP stream
        Given the RTMP stream is started with audio_channel_map/<ptv_file> and "-d 0 -f 0 -l 3000"
        Then The background process was successful
        Then I wait for 10 seconds
        Then I copy the file from the wowza server
        Then I strip the audio from the recorded video file
        Then I expect the output audio channel map of audio_channel_map/output.wav to be "<audio_map_output>"

        Examples:

        | ptv_file                    | audio_map_output |
        | sin-gen-stereo-rtmp.ptv     | 0 1              |
        | sin-gen-amb-wxyz-rtmp.ptv   | 0 1 2 3          |
