# test the web optimization for YouTube

Feature: Output codec
    Scenario: Output codec without audio
        When I launch videostitch-cmd with videoformat01/h264_nvenc.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed
        Then I check videoformat01/output.mp4 faststart with AtomicParsley
        Then I check videoformat01/output.mp4 Atom avcC with AtomicParsley
        Then I check videoformat01/output.mp4 Atom uuid=ffcc8263-f855-4a93-8814-587a02521fdd with AtomicParsley
        Then I check videoformat01/output.mp4 no Atom mp4a with AtomicParsley
        Then I check videoformat01/output.mp4 no Atom SA3D with AtomicParsley
        When I check videoformat01/output.mp4 integrity with avprobe
        Then The video is OK

    Scenario: Output codec with stereo
        When I launch videostitch-cmd with videoformat01/template_aac_128kbs.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed
        Then I check videoformat01/output.mp4 faststart with AtomicParsley
        Then I check videoformat01/output.mp4 Atom uuid=ffcc8263-f855-4a93-8814-587a02521fdd with AtomicParsley
        Then I check videoformat01/output.mp4 Atom mp4a with AtomicParsley
        Then I check videoformat01/output.mp4 no Atom SA3D with AtomicParsley
        When I check videoformat01/output.mp4 integrity with avprobe
        Then The video is OK

    Scenario: Output codec with ambisonic
        When I launch videostitch-cmd with videoformat01/template_ambi_512kbs.ptv and "-d 0 -f 0 -l 10"
        Then I expect the command to succeed
        Then I check videoformat01/output.mp4 faststart with AtomicParsley
        Then I check videoformat01/output.mp4 Atom uuid=ffcc8263-f855-4a93-8814-587a02521fdd with AtomicParsley
        Then I check videoformat01/output.mp4 Atom mp4a with AtomicParsley
        Then I check videoformat01/output.mp4 Atom SA3D with AtomicParsley
        When I check videoformat01/output.mp4 integrity with avprobe
        Then The video is OK


