Feature: RTMP Streaming with bandwidth limitations

    @slow
    Scenario Outline: RTMP encoder dynamic bandwidth
        Given the RTMP stream is started with bandwidth limited to <bandwidth> with videoformat01/<test>.ptv and "-d 0 -f 0 -l 1000"
        Then The background process was successful
        Then I copy the file from the wowza server
        Then I expect the number of frames of the recorded video file to be <frame_order> than <nb_frames>
        And  I check the video effective_bitrate of the recorded video file for 20 seconds is <br_order> than <bitrate>

    Examples:
      | bandwidth | br_order  | bitrate | test                    | frame_order  | nb_frames |
      | 2000      | around    | 2000    | template_rtmp_vbr       | higher       | 945       |
      # packets should be dropped as dynamic bitrate is not set 
      | 200       | higher    | 2100    | template_rtmp_cbr       | lower        | 680       |
      # bit rate should be reduced as dynamic bitrate is set, but output queue not empty 
      | 250       | lower     | 1900    | template_rtmp_vbr       | higher       | 750       |
      | 290       | lower     | 1900    | template_rtmp_nvenc_vbr | higher       | 780       |
      | 295       | lower     | 1900    | template_rtmp_nvenc_cbr | higher       | 780       |

