// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace Testing {

void resetCalibrationTest() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parse("data/calibrated_full.ptv")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(parser->getRoot());

  // Check there is a calibration applied to the pano.
  ENSURE(panoDef.status());
  ENSURE(panoDef->numVideoInputs() == 6);
  ENSURE(panoDef->hasCalibrationControlPoints());
  ENSURE(panoDef->hasCalibrationRigPresets());
  ENSURE(panoDef->hasBeenCalibrated());

  // Reset all the calibration parameters
  panoDef->resetCalibration();

  // Check after reseting
  ENSURE(!panoDef->hasCalibrationControlPoints());
  ENSURE(!panoDef->hasCalibrationRigPresets());
  ENSURE(!panoDef->hasBeenCalibrated());

  std::unique_ptr<VideoStitch::Ptv::Value> pano(panoDef->serialize());

  // Check the serialized json has no calibration elements
  std::stringstream message;
  ENSURE(panoDef->validate(message));
  ENSURE(!pano->has("rig"));
  ENSURE(!pano->has("cameras"));
  ENSURE(pano->has("calibration_control_points")->asList().empty());
}

void checkCalibrationNoGeometries() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parse("data/no_calibration.ptv")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(panoDef.status());
  ENSURE(!panoDef->hasBeenCalibrated());
}

void checkCalibrationTwoGeometries() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parse("data/calibration_only_geometries2.ptv")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(panoDef.status());
  ENSURE(!panoDef->hasBeenCalibrated());
}

void checkCalibrationOneGeometry() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parse("data/calibration_only_geometries1.ptv")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(panoDef.status());
  ENSURE(!panoDef->hasBeenCalibrated());
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::checkCalibrationNoGeometries();
  VideoStitch::Testing::checkCalibrationTwoGeometries();
  VideoStitch::Testing::checkCalibrationOneGeometry();
  VideoStitch::Testing::resetCalibrationTest();
  return 0;
}
