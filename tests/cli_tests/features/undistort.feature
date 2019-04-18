# Processing undistort with the command-line tool

Feature: undistort

    Scenario Outline: undistort command-line
        "Compares first and second inputs"
        When I launch undistort with undistort/ricoh_theta/ricoh_theta.ptv and "<cmdlineflags> -f 0 -l 0 -o ricoh_theta_undistorted.ptv"
        Then I expect the command to succeed
        When I compare undistort/ricoh_theta/vs-out-undistorted-0-0.jpg with undistort/ricoh_theta/vs-out-undistorted-0-0-reference<ref_suffix>.jpg
        Then I expect the comparison error to be less than 0.005
        When I compare undistort/ricoh_theta/vs-out-undistorted-1-0.jpg with undistort/ricoh_theta/vs-out-undistorted-1-0-reference<ref_suffix>.jpg
        Then I expect the comparison error to be less than 0.005

        Examples:
           | cmdlineflags        | ref_suffix |
           |                     | -v2        |
           | --focal 500         | -500-v2    |

    @slow
    Scenario Outline: undistort command-line
        "Compares first and last inputs. Last input has heavy vignetting and white balance applied."
        When I launch undistort with undistort/6_dof_office/6_dof_office.ptv and "<cmdlineflags> -f 0 -l 0 -o 6_dof_office_undistorted.ptv"
        Then I expect the command to succeed
        When I compare undistort/6_dof_office/vs-out-undistorted-0-0.jpg with undistort/6_dof_office/vs-out-undistorted-0-0-reference<ref_suffix>.jpg
        Then I expect the comparison error to be less than 0.005
        When I compare undistort/6_dof_office/vs-out-undistorted-5-0.jpg with undistort/6_dof_office/vs-out-undistorted-5-0-reference<ref_suffix>.jpg
        Then I expect the comparison error to be less than 0.005

        Examples:
           | cmdlineflags                                                                    | ref_suffix         |
           | --reset_rotation --focal 1400                                                   | -ff-opt-rotation   |
           | --reset_rotation --focal 1400 --projection ff_fisheye                           | -ff-rotation       |
           | --reset_rotation --projection ff_fisheye --width 512 --height 1024              | -ff-width-height   |
           | --reset_rotation --focal 500 --projection rectilinear                           | -rect-rotation     |
           | --reset_rotation --focal 500 --projection equirectangular                       | -equirect-rotation |

    @slow
    Scenario: undistort command-line equirect
        "Undistorting an equirect input. Output should be identical to the input"
        When I launch undistort with undistort/bretagne/bretagne_equirect.ptv and "-f 0 -l 0 -o bretagne_equirect_out.ptv"
        Then I expect the command to succeed
        When I compare undistort/bretagne/vs-out-undistorted-0-0.jpg with undistort/bretagne/input_equirect.jpg
        Then I expect the comparison error to be less than 0.005

    @slow
    Scenario: undistort alpha channel output
        "Create a PNG output with an alpha channel that serves as a mask"
        When I launch undistort with undistort/6_dof_office/6_dof_office_alpha.ptv and "--reset_rotation --projection equirectangular --width 2048 --height 1024 -f 0 -l 0 -o 6_dof_office_undistorted_alpha.ptv"
        Then I expect the command to succeed
        When I replace transparency with a red background in undistort/6_dof_office/vs-out-undistorted-3-0.png
        Then I expect the command to succeed
        When I compare undistort/6_dof_office/vs-out-undistorted-3-0.png with undistort/6_dof_office/vs-out-undistorted-3-0-reference-alpha.png
        Then I expect the comparison error to be less than 0.005
