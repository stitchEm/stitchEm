Feature: RTMP Streaming

  Using videostitch-cmd to stitch some local files, and send publish them with RTMP
  to a local test server.

    Scenario: RTMP Publishing
        When I launch videostitch-cmd with videoformat01/rtmp.ptv and "-d 0 -f 0 -l 100"
        Then I expect the command to succeed

    @slow
    Scenario Outline: RTMP encoder bitrate
        Given the RTMP stream is started with videoformat01/template_rtmp_<bitrate_mode>.ptv and "-d 0 -f 0 -l 800"
        Then The background process was successful
        Then I copy the file from the wowza server
        Then I check the video effective_bitrate of the recorded video file for 20 seconds is <bitrate> with precision of <error>

    Examples:
      | bitrate |   error  | bitrate_mode |
      | 2000    |   0.08   | cbr          |
      | 2000    |   0.08   | vbr          |

