# Processing epipolar

@epipolar
@slow

Feature: epipolar
    Scenario Outline: epipolar
        Given I use epipolar.json for epipolar
        When  I launch videostitch-cmd for epipolar with epipolar/scenes/<test>/<test>.ptv and " -d 0 -v 0 "
        Then  I expect the command to succeed
        And   I expect epipolar/scenes/<test>/output_epipolar.json is the same as epipolar/scenes/<test>/output_epipolar_ref.json with 6 digits after the decimal point for float
        When  I compare epipolar/scenes/<test>/input-00.jpg_frame_0.png with epipolar/scenes/<test>/reference-grid-input-00.jpg_frame_0.png
        Then  I expect the comparison error to be less than 0.001
        When  I compare epipolar/scenes/<test>/input-01.jpg_frame_0.png with epipolar/scenes/<test>/reference-grid-input-01.jpg_frame_0.png
        Then  I expect the comparison error to be less than 0.001
        When  I compare epipolar/scenes/<test>/input-02.jpg_frame_0.png with epipolar/scenes/<test>/reference-grid-input-02.jpg_frame_0.png
        Then  I expect the comparison error to be less than 0.001
        When  I compare epipolar/scenes/<test>/input-03.jpg_frame_0.png with epipolar/scenes/<test>/reference-grid-input-03.jpg_frame_0.png
        Then  I expect the comparison error to be less than 0.001
        When  I compare epipolar/scenes/<test>/output_min_stitching_distance.png with epipolar/scenes/<test>/reference-output_min_stitching_distance.png
        Then  I expect the comparison error to be less than 0.001

        Examples:
            | test                |
            | demo_room           |
            | factory_incremental |
