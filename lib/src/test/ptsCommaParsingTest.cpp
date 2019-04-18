// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <fstream>

namespace VideoStitch {
namespace Testing {

void ptsToPanoDefinitionTest() {
  std::string testData = getDataFolder();
  std::string testFile = testData + "/__tmp__pts_test__.ptv";
  {
    std::ofstream ofs(testFile, std::ios_base::out);
    ofs << "# ptGui project file\n\n"
           "#-encoding utf8\n"
           "#-pathseparator \\\n"
           "#-fileversion 48\n"
           "#-previewwidth 1132\n"
           "#-previewheight 566\n"
           "#-vfov 180\n"
           "#-resolution 300\n"
           "#-fixaspect 1\n"
           "#-outputfile \n"
           "#-ccdcrop 1\n"
           "#-hasbeenoptimized 1\n"
           "#-hvcpmode 1\n"
           "#-jpegparameters 100 0\n"
           "#-tiffparameters 8bit packbits alpha_assoc\n"
           "#-psdparameters 8bit packbits layered\n"
           "#-qtvrparameters 800 600 1 1000 70 0 0 -180 180 0 -90 90 90 10 120 1\n"
           "#-honorexiforientation 1\n"
           "#-exrparameters noalpha\n"
           "#-hdroutputhdrblended\n"
           "#-hdroutputtonemapped\n"
           "#-hdrfileformat hdr\n"
           "#-hdrmethod fuse\n"
           "#-hdrpsdparameters float none layered\n"
           "#-tonemapv2settings PTGTM 1 0.5 20 20 0 0 2 1 0.15 0\n"
           "#-fusesettings 0.5 0 0.2 0 0\n"
           "#-cameracurve 3.441228 -0.551975 0.294423 -0.095632 0.072139\n"
           "#-vignettingparams 0.1 0.2 0.3 0.4 0.5\n"
           "#-wbexposure 0 0 0\n"
           "#-pmoptexposuremode auto\n"
           "#-pmoptvignettingmode enabled\n"
           "#-pmoptwbmode disabled\n"
           "#-pmoptflaremode disabled\n"
           "#-pmoptcameracurvemode auto\n"
           "#-exposurecorrection 0\n"
           "#-tonemapldrpanorama 0\n"
           "#-blendweight 100 100 100 100 100 100\n"
           "#-optviewpoint 000000\n"
           "#-useexif0\n"
           "#-batchbuilder_useexif 0\n"
           "#-stitcher ptgui\n"
           "#-blender ptgui\n"
           "#-blenderfeather 0\n"
           "#-optimizer ptgui\n"
           "#-interpolator default\n"
           "#-autocpdone\n"
           "#-imgrotate444444\n"
           "#-cpinactive\n"
           "#-imginactive\n"
           "#-linktoprevious\n"
           "#-previewinactive\n"
           "#-outputcrop 0 1 0 1\n"
           "#-location default\n"
           "#-viewingdirection default\n"
           "#-alignsettings_generatecp 1\n"
           "#-alignsettings_optimize 1\n"
           "#-alignsettings_optimizeprealign 1\n"
           "#-alignsettings_straighten 1\n"
           "#-alignsettings_fit 1\n"
           "#-alignsettings_chooseprojection 1\n"
           "#-alignsettings_setoptimumsize 1\n"
           "#-alignsettings_limitsize 500\n"
           "#-alignsettings_optimizeexposure 0\n"
           "#-hdrsettings_defaultlinkmode nolink\n"
           "#-hdrsettings_donotask 0\n"
           "#-batchsettings_align 0\n"
           "#-batchsettings_stitch 1\n"
           "#-batchsettings_stitchonlyifcontrolpoints 1\n"
           "#-defaultprojectfilenamemode firstsourceimage\n"
           "#-defaultprojectfilename_custom \" Panorama\"\n"
           "#-defaultprojectfoldermode sourcefolder\n"
           "#-defaultprojectfolder_custom "
           "\n"
           "#-defaultpanoramafilenamemode asproject\n"
           "#-defaultpanoramafilename_custom "
           "\n"
           "#-defaultpanoramafoldermode projectfolder\n"
           "#-defaultpanoramafolder_custom "
           "\n"
           "#-userelativesourceimagepaths 1\n"
           "#-optimizeraskreinitialize 1\n"
           "#-applytemplate_lens 1\n"
           "#-applytemplate_imageparams 1\n"
           "#-applytemplate_crop 1\n"
           "#-applytemplate_mask 1\n"
           "#-applytemplate_panoramasettings 1\n"
           "#-applytemplate_projectsettings 1\n"
           "#-applytemplate_optimizer 1\n"
           "#-globalcrop 0 0 0 0 0 0 0 0.5\n"
           "#-theoreticalhfov -1\n"
           "#-rect_compression_x 0\n"
           "#-rect_compression_y 0\n"
           "#-cylindrical_compression_y 0\n"
           "#-transverse_cylindrical_compression_x 0\n"
           "#-vedutismo_compression_x 1\n"
           "#-transverse_vedutismo_compression_y 1\n"
           "#-stereographic_compression 1\n"
           "#-rectifisheye_compression 1\n"
           "# Panorama settings:\n"
           "p w5548 h2774 f2 v360 u0 n\"JPEG g0 q100\"\n"
           "m g0 i0\n"

           "# input images:\n"
           "#-dummyimage\n"
           "# The following line contains a 'dummy image' containing some global parameters for the project\n"
           "o w1 h1 y0 r0 p0 v124.785166312651 a-0.1712794685889779 b0.6799971985994054 c-0.7660565920132166 f3 "
           "d0.0009362105412644209 e-0.0007190110954970641 g0 t0\n"
           "#-imgfile 1920 1440 \"input-00.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "#-imgcrop 0 0 0 0 0 0.0015625 -0.008333333333333333 0.3526041666666667\n"
           "o f3 y100.7017117457532 r45.10757670220181 p-35.01011030227659 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n"
           "#-imgfile 1920 1440 \"input-01.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "o f3 y-77.97401527325749 r-44.1078670901295 p34.59807706840689 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n"
           "#-imgfile 1920 1440 \"input-02.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "o f3 y44.26937125237751 r-44.40647798931266 p33.31125271366329 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n"
           "#-imgfile 1920 1440 \"input-03.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "o f3 y-138.9538028541613 r46.74419464943895 p-33.95299788917947 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n"
           "#-imgfile 1920 1440 \"input-04.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "o f3 y-18.40629128142902 r45.03411964820657 p-36.46970965651491 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n"
           "#-imgfile 1920 1440 \"input-05.jpg\"\n"
           "#-metadata -1 -1 -1 0000-00-00T00:00:00 3*8 0 -1 -1 -1 * * * curve -1 * T *\n"
           "#-viewpoint 0 0 0 0 0\n"
           "#-exposureparams 0 0 0 0\n"
           "o f3 y162.7139922077101 r-44.61102349674985 p35.90588750775342 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0\n";
  }
  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::parseFromPto(testFile);
  ENSURE(panoDef.status());

  // Validate PanoDef
  ENSURE_EQ(6, (int)panoDef->numInputs());
  ENSURE_EQ(5548, (int)panoDef->getWidth());
  ENSURE_EQ(2774, (int)panoDef->getHeight());
  ENSURE(Core::PanoProjection::Equirectangular == panoDef->getProjection());
  ENSURE_APPROX_EQ(360.0, panoDef->getHFOV(), 0.00001);

  for (readerid_t inputId = 0; inputId < panoDef->numInputs(); ++inputId) {
    const Core::InputDefinition* input = &panoDef->getInput(inputId);

    ENSURE_EQ(1920, (int)input->getWidth());
    ENSURE_EQ(1440, (int)input->getHeight());
    ENSURE_EQ(0, (int)input->getCropLeft());
    ENSURE_EQ(1920, (int)input->getCropRight());
    ENSURE_EQ(0, (int)input->getCropTop());
    ENSURE_EQ(1440, (int)input->getCropBottom());
    ENSURE_EQ((int)Core::InputDefinition::Format::FullFrameFisheye, (int)input->getFormat());
    ENSURE_APPROX_EQ(3.441228, input->getEmorA(), 0.00001);
    ENSURE_APPROX_EQ(-0.551975, input->getEmorB(), 0.00001);
    ENSURE_APPROX_EQ(0.294423, input->getEmorC(), 0.00001);
    ENSURE_APPROX_EQ(-0.095632, input->getEmorD(), 0.00001);
    ENSURE_APPROX_EQ(0.072139, input->getEmorE(), 0.00001);
    ENSURE_APPROX_EQ(1.0, input->getGamma(), 0.00001);
    ENSURE_APPROX_EQ(-0.1712794685889779, input->getGeometries().at(0).getDistortA(), 0.00001);
    ENSURE_APPROX_EQ(0.6799971985994054, input->getGeometries().at(0).getDistortB(), 0.00001);
    ENSURE_APPROX_EQ(-0.7660565920132166, input->getGeometries().at(0).getDistortC(), 0.00001);
    ENSURE_APPROX_EQ(0.0009362105412644209, input->getGeometries().at(0).getCenterX(), 0.00001);
    ENSURE_APPROX_EQ(-0.0007190110954970641, input->getGeometries().at(0).getCenterY(), 0.00001);
    ENSURE_APPROX_EQ(1.0, input->getVignettingCoeff0(), 0.00001);
    ENSURE_APPROX_EQ(0.1, input->getVignettingCoeff1(), 0.00001);
    ENSURE_APPROX_EQ(0.2, input->getVignettingCoeff2(), 0.00001);
    ENSURE_APPROX_EQ(0.3, input->getVignettingCoeff3(), 0.00001);
    ENSURE_APPROX_EQ(0.4, input->getVignettingCenterX(), 0.00001);
    ENSURE_APPROX_EQ(0.5, input->getVignettingCenterY(), 0.00001);
    ENSURE_EQ((int)Core::InputDefinition::PhotoResponse::InvEmorResponse, (int)input->getPhotoResponse());
    ENSURE_APPROX_EQ(124.785166312651, input->getGeometries().at(0).getHorizontalFocal(), 0.00001);
    ENSURE_EQ(false, input->hasCroppedArea());
  }

  ENSURE_APPROX_EQ(100.7017117457532, panoDef->getInput(0).getGeometries().at(0).getYaw(), 0.00001);
  ENSURE_APPROX_EQ(-77.97401527325749, panoDef->getInput(1).getGeometries().at(0).getYaw(), 0.00001);
  ENSURE_APPROX_EQ(44.26937125237751, panoDef->getInput(2).getGeometries().at(0).getYaw(), 0.00001);
  ENSURE_APPROX_EQ(-138.9538028541613, panoDef->getInput(3).getGeometries().at(0).getYaw(), 0.00001);
  ENSURE_APPROX_EQ(-18.40629128142902, panoDef->getInput(4).getGeometries().at(0).getYaw(), 0.00001);
  ENSURE_APPROX_EQ(162.7139922077101, panoDef->getInput(5).getGeometries().at(0).getYaw(), 0.00001);

  ENSURE_APPROX_EQ(45.10757670220181, panoDef->getInput(0).getGeometries().at(0).getRoll(), 0.00001);
  ENSURE_APPROX_EQ(-44.1078670901295, panoDef->getInput(1).getGeometries().at(0).getRoll(), 0.00001);
  ENSURE_APPROX_EQ(-44.40647798931266, panoDef->getInput(2).getGeometries().at(0).getRoll(), 0.00001);
  ENSURE_APPROX_EQ(46.74419464943895, panoDef->getInput(3).getGeometries().at(0).getRoll(), 0.00001);
  ENSURE_APPROX_EQ(45.03411964820657, panoDef->getInput(4).getGeometries().at(0).getRoll(), 0.00001);
  ENSURE_APPROX_EQ(-44.61102349674985, panoDef->getInput(5).getGeometries().at(0).getRoll(), 0.00001);

  ENSURE_APPROX_EQ(-35.01011030227659, panoDef->getInput(0).getGeometries().at(0).getPitch(), 0.00001);
  ENSURE_APPROX_EQ(34.59807706840689, panoDef->getInput(1).getGeometries().at(0).getPitch(), 0.00001);
  ENSURE_APPROX_EQ(33.31125271366329, panoDef->getInput(2).getGeometries().at(0).getPitch(), 0.00001);
  ENSURE_APPROX_EQ(-33.95299788917947, panoDef->getInput(3).getGeometries().at(0).getPitch(), 0.00001);
  ENSURE_APPROX_EQ(-36.46970965651491, panoDef->getInput(4).getGeometries().at(0).getPitch(), 0.00001);
  ENSURE_APPROX_EQ(35.90588750775342, panoDef->getInput(5).getGeometries().at(0).getPitch(), 0.00001);
}

void ptsAppliedToPanoDefinitionTest() {
  const std::string inputPanoJsonSerialized = R"({
  "width" : 4096,
  "height" : 2048,
  "hfov" : 360,
  "proj" : "equirectangular",
  "global_orientation" : [
                          1,
                          0,
                          0,
                          0
                          ],
  "stabilization" : [
                     1,
                     0,
                     0,
                     0
                     ],
  "ev" : 0,
  "red_corr" : 1,
  "green_corr" : 1,
  "blue_corr" : 1,
  "calibration_cost" : -757.557,
  "inputs" : [
              {
              "reader_config" : "input-00.jpg",
              "group" : 0,
              "width" : 1920,
              "height" : 1080,
              "mask_data" : "",
              "no_delete_masked_pixels" : false,
              "proj" : "circular_fisheye",
              "crop_left" : -21,
              "crop_right" : 1941,
              "crop_top" : -441,
              "crop_bottom" : 1521,
              "ev" : 0,
              "red_corr" : 1,
              "green_corr" : 1,
              "blue_corr" : 1,
              "response" : "emor",
              "useMeterDistortion" : true,
              "emor_a" : 0,
              "emor_b" : 0,
              "emor_c" : 0,
              "emor_d" : 0,
              "emor_e" : 0,
              "gamma" : 1,
              "vign_a" : 1,
              "vign_b" : 0,
              "vign_c" : 0,
              "vign_d" : 0,
              "vign_x" : 0,
              "vign_y" : 0,
              "frame_offset" : 37,
              "synchro_cost" : -1,
              "stack_order" : 0,
              "geometries" : [
                              0,
                              0,
                              {
                              "yaw" : 143.221,
                              "pitch" : 6.89742,
                              "roll" : -91.0976,
                              "center_x" : -29.0843,
                              "center_y" : 7.17565,
                              "distort_a" : 0,
                              "distort_b" : -0.0744755,
                              "distort_c" : 0,
                              "horizontalFocal" : 639.513
                              }
                              ],
              "video_enabled" : true,
              "audio_enabled" : false
              },
              {
              "reader_config" : "input-01.jpg",
              "group" : 0,
              "width" : 1920,
              "height" : 1080,
              "mask_data" : "",
              "no_delete_masked_pixels" : false,
              "proj" : "circular_fisheye",
              "crop_left" : -21,
              "crop_right" : 1941,
              "crop_top" : -441,
              "crop_bottom" : 1521,
              "ev" : 0,
              "red_corr" : 1,
              "green_corr" : 1,
              "blue_corr" : 1,
              "response" : "emor",
              "useMeterDistortion" : true,
              "emor_a" : 0,
              "emor_b" : 0,
              "emor_c" : 0,
              "emor_d" : 0,
              "emor_e" : 0,
              "gamma" : 1,
              "vign_a" : 1,
              "vign_b" : 0,
              "vign_c" : 0,
              "vign_d" : 0,
              "vign_x" : 0,
              "vign_y" : 0,
              "frame_offset" : 0,
              "synchro_cost" : -1,
              "stack_order" : 0,
              "geometries" : [
                              0,
                              0,
                              {
                              "yaw" : 69.6377,
                              "pitch" : 0.912142,
                              "roll" : -89.1236,
                              "center_x" : 5.51802,
                              "center_y" : -6.6365,
                              "distort_a" : 0,
                              "distort_b" : -0.0969694,
                              "distort_c" : 0,
                              "horizontalFocal" : 630.208
                              }
                              ],
              "video_enabled" : true,
              "audio_enabled" : false
              },
              {
              "reader_config" : "input-02.jpg",
              "group" : 0,
              "width" : 1920,
              "height" : 1080,
              "mask_data" : "",
              "no_delete_masked_pixels" : false,
              "proj" : "circular_fisheye",
              "crop_left" : -21,
              "crop_right" : 1941,
              "crop_top" : -441,
              "crop_bottom" : 1521,
              "ev" : 0,
              "red_corr" : 1,
              "green_corr" : 1,
              "blue_corr" : 1,
              "response" : "emor",
              "useMeterDistortion" : true,
              "emor_a" : 0,
              "emor_b" : 0,
              "emor_c" : 0,
              "emor_d" : 0,
              "emor_e" : 0,
              "gamma" : 1,
              "vign_a" : 1,
              "vign_b" : 0,
              "vign_c" : 0,
              "vign_d" : 0,
              "vign_x" : 0,
              "vign_y" : 0,
              "frame_offset" : 141,
              "synchro_cost" : -1,
              "stack_order" : 0,
              "geometries" : [
                              0,
                              0,
                              {
                              "yaw" : -4.75041,
                              "pitch" : -0.752223,
                              "roll" : -88.2187,
                              "center_x" : -34.0097,
                              "center_y" : -20.476,
                              "distort_a" : 0,
                              "distort_b" : -0.0598133,
                              "distort_c" : 0,
                              "horizontalFocal" : 639.372
                              }
                              ],
              "video_enabled" : true,
              "audio_enabled" : false
              },
              {
              "reader_config" : "input-03.jpg",
              "group" : 0,
              "width" : 1920,
              "height" : 1080,
              "mask_data" : "",
              "no_delete_masked_pixels" : false,
              "proj" : "circular_fisheye",
              "crop_left" : -21,
              "crop_right" : 1941,
              "crop_top" : -441,
              "crop_bottom" : 1521,
              "ev" : 0,
              "red_corr" : 1,
              "green_corr" : 1,
              "blue_corr" : 1,
              "response" : "emor",
              "useMeterDistortion" : true,
              "emor_a" : 0,
              "emor_b" : 0,
              "emor_c" : 0,
              "emor_d" : 0,
              "emor_e" : 0,
              "gamma" : 1,
              "vign_a" : 1,
              "vign_b" : 0,
              "vign_c" : 0,
              "vign_d" : 0,
              "vign_x" : 0,
              "vign_y" : 0,
              "frame_offset" : 182,
              "synchro_cost" : -1,
              "stack_order" : 0,
              "geometries" : [
                              0,
                              0,
                              {
                              "yaw" : -74.2756,
                              "pitch" : 2.61056,
                              "roll" : -89.8794,
                              "center_x" : -19.6266,
                              "center_y" : 3.34039,
                              "distort_a" : 0,
                              "distort_b" : -0.0745318,
                              "distort_c" : 0,
                              "horizontalFocal" : 628.602
                              }
                              ],
              "video_enabled" : true,
              "audio_enabled" : false
              },
              {
              "reader_config" : "input-04.jpg",
              "group" : 0,
              "width" : 1920,
              "height" : 1080,
              "mask_data" : "",
              "no_delete_masked_pixels" : false,
              "proj" : "circular_fisheye",
              "crop_left" : -21,
              "crop_right" : 1941,
              "crop_top" : -441,
              "crop_bottom" : 1521,
              "ev" : 0,
              "red_corr" : 1,
              "green_corr" : 1,
              "blue_corr" : 1,
              "response" : "emor",
              "useMeterDistortion" : true,
              "emor_a" : 0,
              "emor_b" : 0,
              "emor_c" : 0,
              "emor_d" : 0,
              "emor_e" : 0,
              "gamma" : 1,
              "vign_a" : 1,
              "vign_b" : 0,
              "vign_c" : 0,
              "vign_d" : 0,
              "vign_x" : 0,
              "vign_y" : 0,
              "frame_offset" : 277,
              "synchro_cost" : -1,
              "stack_order" : 0,
              "geometries" : [
                              0,
                              0,
                              {
                              "yaw" : -141.872,
                              "pitch" : 2.62554,
                              "roll" : -91.3031,
                              "center_x" : 4.37194,
                              "center_y" : 29.526,
                              "distort_a" : 0,
                              "distort_b" : -0.0879253,
                              "distort_c" : 0,
                              "horizontalFocal" : 627.042
                              }
                              ],
              "video_enabled" : true,
              "audio_enabled" : false
              }
              ],
  "merger_mask" : {
    "width" : 0,
    "height" : 0,
    "enable" : false,
    "interpolationEnabled" : false,
    "inputScaleFactor" : 2,
    "masks" : []
  }
  })";

  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser->parseData(inputPanoJsonSerialized)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> inputPanoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(inputPanoDef.status());

  Potential<Core::PanoDefinition> panoDef =
      Core::PanoDefinition::parseFromPto("data/pts/input_panorama.pts", inputPanoDef.object());
  ENSURE(panoDef.status());

  const std::string refPanoJsonSerialized = R"({
    "width" : 4096,
    "height" : 2048,
    "hfov" : 360,
    "proj" : "equirectangular",
    "global_orientation" : [
                            1,
                            0,
                            0,
                            0
                            ],
    "stabilization" : [
                       1,
                       0,
                       0,
                       0
                       ],
    "ev" : 0,
    "red_corr" : 1,
    "green_corr" : 1,
    "blue_corr" : 1,
    "calibration_cost" : -757.557,
    "inputs" : [
                {
                "reader_config" : "input-00.jpg",
                "group" : 0,
                "width" : 1920,
                "height" : 1080,
                "mask_data" : "",
                "no_delete_masked_pixels" : false,
                "proj" : "circular_fisheye",
                "crop_left" : -21,
                "crop_right" : 1941,
                "crop_top" : -441,
                "crop_bottom" : 1521,
                "ev" : 0,
                "red_corr" : 1,
                "green_corr" : 1,
                "blue_corr" : 1,
                "response" : "emor",
                "emor_a" : 0,
                "emor_b" : 0,
                "emor_c" : 0,
                "emor_d" : 0,
                "emor_e" : 0,
                "gamma" : 1,
                "vign_a" : 1,
                "vign_b" : 0,
                "vign_c" : 0,
                "vign_d" : 0,
                "vign_x" : 0,
                "vign_y" : 0,
                "frame_offset" : 37,
                "synchro_cost" : -1,
                "stack_order" : 0,
                "geometries" : [
                                0,
                                0,
                                {
                                "yaw" : 143.221,
                                "pitch" : 6.89742,
                                "roll" : -91.0976,
                                "center_x" : -29.0843,
                                "center_y" : 7.17565,
                                "distort_a" : 0,
                                "distort_b" : -0.0744755,
                                "distort_c" : 0,
                                "horizontalFocal" : 639.513
                                }
                                ],
                "video_enabled" : true,
                "audio_enabled" : false
                },
                {
                "reader_config" : "input-01.jpg",
                "group" : 0,
                "width" : 1920,
                "height" : 1080,
                "mask_data" : "",
                "no_delete_masked_pixels" : false,
                "proj" : "circular_fisheye",
                "crop_left" : -21,
                "crop_right" : 1941,
                "crop_top" : -441,
                "crop_bottom" : 1521,
                "ev" : 0,
                "red_corr" : 1,
                "green_corr" : 1,
                "blue_corr" : 1,
                "response" : "emor",
                "emor_a" : 0,
                "emor_b" : 0,
                "emor_c" : 0,
                "emor_d" : 0,
                "emor_e" : 0,
                "gamma" : 1,
                "vign_a" : 1,
                "vign_b" : 0,
                "vign_c" : 0,
                "vign_d" : 0,
                "vign_x" : 0,
                "vign_y" : 0,
                "frame_offset" : 0,
                "synchro_cost" : -1,
                "stack_order" : 0,
                "geometries" : [
                                0,
                                0,
                                {
                                "yaw" : 69.6377,
                                "pitch" : 0.912142,
                                "roll" : -89.1236,
                                "center_x" : 5.51802,
                                "center_y" : -6.6365,
                                "distort_a" : 0,
                                "distort_b" : -0.0969694,
                                "distort_c" : 0,
                                "horizontalFocal" : 630.208
                                }
                                ],
                "video_enabled" : true,
                "audio_enabled" : false
                },
                {
                "reader_config" : "input-02.jpg",
                "group" : 0,
                "width" : 1920,
                "height" : 1080,
                "mask_data" : "",
                "no_delete_masked_pixels" : false,
                "proj" : "circular_fisheye",
                "crop_left" : -21,
                "crop_right" : 1941,
                "crop_top" : -441,
                "crop_bottom" : 1521,
                "ev" : 0,
                "red_corr" : 1,
                "green_corr" : 1,
                "blue_corr" : 1,
                "response" : "emor",
                "emor_a" : 0,
                "emor_b" : 0,
                "emor_c" : 0,
                "emor_d" : 0,
                "emor_e" : 0,
                "gamma" : 1,
                "vign_a" : 1,
                "vign_b" : 0,
                "vign_c" : 0,
                "vign_d" : 0,
                "vign_x" : 0,
                "vign_y" : 0,
                "frame_offset" : 141,
                "synchro_cost" : -1,
                "stack_order" : 0,
                "geometries" : [
                                0,
                                0,
                                {
                                "yaw" : -4.75041,
                                "pitch" : -0.752223,
                                "roll" : -88.2187,
                                "center_x" : -34.0097,
                                "center_y" : -20.476,
                                "distort_a" : 0,
                                "distort_b" : -0.0598133,
                                "distort_c" : 0,
                                "horizontalFocal" : 639.372
                                }
                                ],
                "video_enabled" : true,
                "audio_enabled" : false
                },
                {
                "reader_config" : "input-03.jpg",
                "group" : 0,
                "width" : 1920,
                "height" : 1080,
                "mask_data" : "",
                "no_delete_masked_pixels" : false,
                "proj" : "circular_fisheye",
                "crop_left" : -21,
                "crop_right" : 1941,
                "crop_top" : -441,
                "crop_bottom" : 1521,
                "ev" : 0,
                "red_corr" : 1,
                "green_corr" : 1,
                "blue_corr" : 1,
                "response" : "emor",
                "emor_a" : 0,
                "emor_b" : 0,
                "emor_c" : 0,
                "emor_d" : 0,
                "emor_e" : 0,
                "gamma" : 1,
                "vign_a" : 1,
                "vign_b" : 0,
                "vign_c" : 0,
                "vign_d" : 0,
                "vign_x" : 0,
                "vign_y" : 0,
                "frame_offset" : 182,
                "synchro_cost" : -1,
                "stack_order" : 0,
                "geometries" : [
                                0,
                                0,
                                {
                                "yaw" : -74.2756,
                                "pitch" : 2.61056,
                                "roll" : -89.8794,
                                "center_x" : -19.6266,
                                "center_y" : 3.34039,
                                "distort_a" : 0,
                                "distort_b" : -0.0745318,
                                "distort_c" : 0,
                                "horizontalFocal" : 628.602
                                }
                                ],
                "video_enabled" : true,
                "audio_enabled" : false
                },
                {
                "reader_config" : "input-04.jpg",
                "group" : 0,
                "width" : 1920,
                "height" : 1080,
                "mask_data" : "",
                "no_delete_masked_pixels" : false,
                "proj" : "circular_fisheye",
                "crop_left" : -21,
                "crop_right" : 1941,
                "crop_top" : -441,
                "crop_bottom" : 1521,
                "ev" : 0,
                "red_corr" : 1,
                "green_corr" : 1,
                "blue_corr" : 1,
                "response" : "emor",
                "emor_a" : 0,
                "emor_b" : 0,
                "emor_c" : 0,
                "emor_d" : 0,
                "emor_e" : 0,
                "gamma" : 1,
                "vign_a" : 1,
                "vign_b" : 0,
                "vign_c" : 0,
                "vign_d" : 0,
                "vign_x" : 0,
                "vign_y" : 0,
                "frame_offset" : 277,
                "synchro_cost" : -1,
                "stack_order" : 0,
                "geometries" : [
                                0,
                                0,
                                {
                                "yaw" : -141.872,
                                "pitch" : 2.62554,
                                "roll" : -91.3031,
                                "center_x" : 4.37194,
                                "center_y" : 29.526,
                                "distort_a" : 0,
                                "distort_b" : -0.0879253,
                                "distort_c" : 0,
                                "horizontalFocal" : 627.042
                                }
                                ],
                "video_enabled" : true,
                "audio_enabled" : false
                }
                ],
    "merger_mask" : {
      "width" : 0,
      "height" : 0,
      "enable" : false,
      "interpolationEnabled" : false,
      "inputScaleFactor" : 2,
      "masks" : []
    }
  })";

  Potential<Ptv::Parser> refParser = Ptv::Parser::create();
  if (!refParser->parseData(refPanoJsonSerialized)) {
    std::cerr << refParser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> refPanoDef = Core::PanoDefinition::create(refParser->getRoot());
  ENSURE(refPanoDef.status());

  // Validate PanoDef
  ENSURE_EQ(refPanoDef->numInputs(), panoDef->numInputs());
  ENSURE_EQ(refPanoDef->getWidth(), panoDef->getWidth());
  ENSURE_EQ(refPanoDef->getHeight(), panoDef->getHeight());
  ENSURE(refPanoDef->getProjection() == panoDef->getProjection());
  ENSURE_APPROX_EQ(refPanoDef->getHFOV(), panoDef->getHFOV(), 0.00001);

  for (auto inputId = 0; inputId < int(refPanoDef->numInputs()); ++inputId) {
    const Core::InputDefinition* input = &panoDef->getInput(inputId);
    const Core::InputDefinition* refInput = &refPanoDef->getInput(inputId);

    ENSURE_EQ(input->getWidth(), refInput->getWidth());
    ENSURE_EQ(input->getHeight(), refInput->getHeight());
    ENSURE_EQ(input->getCropLeft(), refInput->getCropLeft());
    ENSURE_EQ(input->getCropRight(), refInput->getCropRight());
    ENSURE_EQ(input->getCropTop(), refInput->getCropTop());
    ENSURE_EQ(input->getCropBottom(), refInput->getCropBottom());
    ENSURE_EQ((int)input->getFormat(), (int)refInput->getFormat());
    ENSURE_APPROX_EQ(input->getEmorA(), refInput->getEmorA(), 0.00001);
    ENSURE_APPROX_EQ(input->getEmorB(), refInput->getEmorB(), 0.00001);
    ENSURE_APPROX_EQ(input->getEmorC(), refInput->getEmorC(), 0.00001);
    ENSURE_APPROX_EQ(input->getEmorD(), refInput->getEmorD(), 0.00001);
    ENSURE_APPROX_EQ(input->getEmorE(), refInput->getEmorE(), 0.00001);
    ENSURE_APPROX_EQ(input->getGamma(), refInput->getGamma(), 0.00001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getDistortA(), refInput->getGeometries().at(0).getDistortA(),
                     0.00001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getDistortB(), refInput->getGeometries().at(0).getDistortB(),
                     0.00001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getDistortC(), refInput->getGeometries().at(0).getDistortC(),
                     0.00001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getCenterX(), refInput->getGeometries().at(0).getCenterX(), 0.0001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getCenterY(), refInput->getGeometries().at(0).getCenterY(), 0.0001);
    ENSURE_APPROX_EQ(input->getVignettingCoeff0(), refInput->getVignettingCoeff0(), 0.00001);
    ENSURE_APPROX_EQ(input->getVignettingCoeff1(), refInput->getVignettingCoeff1(), 0.00001);
    ENSURE_APPROX_EQ(input->getVignettingCoeff2(), refInput->getVignettingCoeff2(), 0.00001);
    ENSURE_APPROX_EQ(input->getVignettingCoeff3(), refInput->getVignettingCoeff3(), 0.00001);
    ENSURE_APPROX_EQ(input->getVignettingCenterX(), refInput->getVignettingCenterX(), 0.00001);
    ENSURE_APPROX_EQ(input->getVignettingCenterY(), refInput->getVignettingCenterY(), 0.00001);
    ENSURE_EQ((int)input->getPhotoResponse(), (int)refInput->getPhotoResponse());
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getHorizontalFocal(),
                     refInput->getGeometries().at(0).getHorizontalFocal(), 0.001);
    ENSURE_EQ(input->hasCroppedArea(), refInput->hasCroppedArea());
    ENSURE_EQ(input->getUseMeterDistortion(), refInput->getUseMeterDistortion());

    ENSURE_APPROX_EQ(input->getGeometries().at(0).getYaw(), refInput->getGeometries().at(0).getYaw(), 0.001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getPitch(), refInput->getGeometries().at(0).getPitch(), 0.001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getRoll(), refInput->getGeometries().at(0).getRoll(), 0.001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getTranslationX(), refInput->getGeometries().at(0).getTranslationX(),
                     0.001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getTranslationY(), refInput->getGeometries().at(0).getTranslationY(),
                     0.001);
    ENSURE_APPROX_EQ(input->getGeometries().at(0).getTranslationZ(), refInput->getGeometries().at(0).getTranslationZ(),
                     0.001);
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  /*Test with comma decimation separator*/
#ifdef _MSC_VER
  std::locale::global(std::locale("French_France.1252"));
#else
  std::locale::global(std::locale("fr_FR.UTF-8"));
#endif

  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ptsToPanoDefinitionTest();
  VideoStitch::Testing::ptsAppliedToPanoDefinitionTest();
  return 0;
}
