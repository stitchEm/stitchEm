// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Testing {

std::unique_ptr<Core::PanoDefinition> getPanoDef(std::string filename) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  std::cout << "Parsing" << std::endl;
  if (!parser->parse(filename)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
    return nullptr;
  }

  Core::PanoDefinition *panoDef = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(panoDef);
  return std::unique_ptr<Core::PanoDefinition>(panoDef);
}

void calibrationWithTranslationTest() {
  auto panoDef = getPanoDef("data/AQ1610012338_pano.ptv");

  ENSURE_EQ(panoDef->numVideoInputs(), 4);
  ENSURE(panoDef->hasTranslations());
  ENSURE_APPROX_EQ(panoDef->computeMinimumRigSphereRadius(), 0.037255, 0.000001);
}

void noTranslationTest() {
  auto panoDef = getPanoDef("data/calibrated_full.ptv");

  ENSURE_EQ(panoDef->numVideoInputs(), 6);
  ENSURE(panoDef->hasBeenCalibrated());

  ENSURE(!panoDef->hasTranslations());
  ENSURE_EQ(panoDef->computeMinimumRigSphereRadius(), 0.);
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::calibrationWithTranslationTest();
  VideoStitch::Testing::noTranslationTest();
  return 0;
}
