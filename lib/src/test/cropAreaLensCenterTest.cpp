// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Testing {

void fullTest() {
  const std::string jsonSerialized = R"({
  "blue_corr": 1,
  "ev": 0,
  "global_orientation": [
                         1,
                         0,
                         0,
                         0
                         ],
  "green_corr": 1,
  "height": 1024,
  "hfov": 360,
  "inputs": [
             {
             "audio_enabled": false,
             "blue_corr": 1,
             "crop_bottom": 1774,
             "crop_left": -69,
             "crop_right": 2015,
             "crop_top": -310,
             "emor_a": 0,
             "emor_b": 0,
             "emor_c": 0,
             "emor_d": 0,
             "emor_e": 0,
             "ev": 0,
             "frame_offset": 0,
             "gamma": 1,
             "geometries": [
                            0,
                            0,
                            {
                            "center_x": -4.21,
                            "center_y": -9.51,
                            "distort_a": 0.1836,
                            "distort_b": -0.5453,
                            "distort_c": -0.0271,
                            "horizontalFocal": 1067.34,
                            "pitch": -18.0,
                            "roll": -90.0,
                            "translation_x": 0.0101,
                            "translation_y": 0.0086,
                            "translation_z": 0.0315,
                            "verticalFocal": 1059.89,
                            "yaw": 0.0
                            }
                            ],
             "green_corr": 1,
             "group": 0,
             "height": 1440,
             "mask_data": "",
             "no_delete_masked_pixels": false,
             "proj": "circular_fisheye_opt",
             "reader_config": "0_0.mp4",
             "red_corr": 1,
             "response": "emor",
             "stack_order": 0,
             "synchro_cost": -1,
             "useMeterDistortion": true,
             "video_enabled": true,
             "vign_a": 1,
             "vign_b": 0,
             "vign_c": 0,
             "vign_d": 0,
             "vign_x": 0,
             "vign_y": 0,
             "width": 1920
             },
             {
             "audio_enabled": false,
             "blue_corr": 1,
             "crop_bottom": 1796,
             "crop_left": -82,
             "crop_right": 2000,
             "crop_top": -286,
             "emor_a": 0,
             "emor_b": 0,
             "emor_c": 0,
             "emor_d": 0,
             "emor_e": 0,
             "ev": 0,
             "frame_offset": 0,
             "gamma": 1,
             "geometries": [
                            0,
                            0,
                            {
                            "center_x": -3.41,
                            "center_y": -1.94,
                            "distort_a": 0.2038,
                            "distort_b": -0.5755,
                            "distort_c": -0.0298,
                            "horizontalFocal": 1063.89,
                            "pitch": -18.41,
                            "roll": -90.26,
                            "translation_x": -0.0326,
                            "translation_y": -0.0072,
                            "translation_z": -0.0941,
                            "verticalFocal": 1056.56,
                            "yaw": 179.03
                            }
                            ],
             "green_corr": 1,
             "group": 0,
             "height": 1440,
             "mask_data": "",
             "no_delete_masked_pixels": false,
             "proj": "circular_fisheye_opt",
             "reader_config": "0_1.mp4",
             "red_corr": 1,
             "response": "emor",
             "stack_order": 0,
             "synchro_cost": -1,
             "useMeterDistortion": true,
             "video_enabled": true,
             "vign_a": 1,
             "vign_b": 0,
             "vign_c": 0,
             "vign_d": 0,
             "vign_x": 0,
             "vign_y": 0,
             "width": 1920
             },
             {
             "audio_enabled": false,
             "blue_corr": 1,
             "crop_bottom": 1729,
             "crop_left": -103,
             "crop_right": 1977,
             "crop_top": -351,
             "emor_a": 0,
             "emor_b": 0,
             "emor_c": 0,
             "emor_d": 0,
             "emor_e": 0,
             "ev": 0,
             "frame_offset": 0,
             "gamma": 1,
             "geometries": [
                            0,
                            0,
                            {
                            "center_x": -5.79,
                            "center_y": 0.07,
                            "distort_a": 0.2046,
                            "distort_b": -0.5795,
                            "distort_c": -0.0126,
                            "horizontalFocal": 1063.46,
                            "pitch": 18.18,
                            "roll": 89.61,
                            "translation_x": -0.0076,
                            "translation_y": -0.0682,
                            "translation_z": -0.0414,
                            "verticalFocal": 1056.27,
                            "yaw": -90.41
                            }
                            ],
             "green_corr": 1,
             "group": 0,
             "height": 1440,
             "mask_data": "",
             "no_delete_masked_pixels": false,
             "proj": "circular_fisheye_opt",
             "reader_config": "1_0.mp4",
             "red_corr": 1,
             "response": "emor",
             "stack_order": 0,
             "synchro_cost": -1,
             "useMeterDistortion": true,
             "video_enabled": true,
             "vign_a": 1,
             "vign_b": 0,
             "vign_c": 0,
             "vign_d": 0,
             "vign_x": 0,
             "vign_y": 0,
             "width": 1920
             },
             {
             "audio_enabled": false,
             "blue_corr": 1,
             "crop_bottom": 1758,
             "crop_left": -134,
             "crop_right": 1994,
             "crop_top": -370,
             "emor_a": 0,
             "emor_b": 0,
             "emor_c": 0,
             "emor_d": 0,
             "emor_e": 0,
             "ev": 0,
             "frame_offset": 0,
             "gamma": 1,
             "geometries": [
                            0,
                            0,
                            {
                            "center_x": 21.98,
                            "center_y": 6.12,
                            "distort_a": 0.2088,
                            "distort_b": -0.5768,
                            "distort_c": -0.0174,
                            "horizontalFocal": 1068.35,
                            "pitch": 17.75,
                            "roll": 90.38,
                            "translation_x": -0.0042,
                            "translation_y": 0.066,
                            "translation_z": -0.0285,
                            "verticalFocal": 1060.87,
                            "yaw": 90.62
                            }
                            ],
             "green_corr": 1,
             "group": 0,
             "height": 1440,
             "mask_data": "",
             "no_delete_masked_pixels": false,
             "proj": "circular_fisheye_opt",
             "reader_config": "1_1.mp4",
             "red_corr": 1,
             "response": "emor",
             "stack_order": 0,
             "synchro_cost": -1,
             "useMeterDistortion": true,
             "video_enabled": true,
             "vign_a": 1,
             "vign_b": 0,
             "vign_c": 0,
             "vign_d": 0,
             "vign_x": 0,
             "vign_y": 0,
             "width": 1920
             }
             ],
  "merger_mask": {
    "enable": false,
    "frameId": -1,
    "height": 0,
    "input_index_count": 0,
    "width": 0
  },
  "proj": "equirectangular",
  "red_corr": 1,
  "spherescale": 1.5,
  "stabilization": [
                    1,
                    0,
                    0,
                    0
                    ],
  "width": 2048

})";

  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parseData(jsonSerialized)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(panoDef.status());

  // Test cropping on first input only
  Core::InputDefinition& firstInput = panoDef->getVideoInput(0);

  ENSURE_EQ(true, firstInput.hasCroppedArea());

  ENSURE_APPROX_EQ(-4.21, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-9.5099999999999997, firstInput.getGeometries().at(0).getCenterY(), 0.00001);

  std::cout << "Testing setCropLeft()" << std::endl;
  firstInput.setCropLeft(-200);
  ENSURE_APPROX_EQ(61.29, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-9.5099999999999997, firstInput.getGeometries().at(0).getCenterY(), 0.00001);

  std::cout << "Testing setCropRight()" << std::endl;
  firstInput.setCropRight(2048);
  ENSURE_APPROX_EQ(44.79, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-9.5099999999999997, firstInput.getGeometries().at(0).getCenterY(), 0.00001);

  std::cout << "Testing setCropTop()" << std::endl;
  firstInput.setCropTop(-100);
  ENSURE_APPROX_EQ(44.79, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-114.51, firstInput.getGeometries().at(0).getCenterY(), 0.00001);

  std::cout << "Testing setCropBottom()" << std::endl;
  firstInput.setCropBottom(2048);
  ENSURE_APPROX_EQ(44.79, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-251.51, firstInput.getGeometries().at(0).getCenterY(), 0.00001);

  std::cout << "Testing setCrop()" << std::endl;
  firstInput.setCrop(-50, 2000, -49, 1920);
  ENSURE_APPROX_EQ(-6.21, firstInput.getGeometries().at(0).getCenterX(), 0.00001);
  ENSURE_APPROX_EQ(-213.01, firstInput.getGeometries().at(0).getCenterY(), 0.00001);
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::fullTest();
  return 0;
}
