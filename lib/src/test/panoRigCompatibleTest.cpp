// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/rigDef.hpp"

#include <memory>

namespace VideoStitch {
namespace Testing {

void rigPanoTest() {
  // Parse and create the pano
  Potential<Ptv::Parser> panoParser = Ptv::Parser::create();
  std::cout << "Parsing pano" << std::endl;
  if (!panoParser->parse("data/rig_pano/pano_6_1920x1080.ptv")) {
    std::cerr << panoParser->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  const Potential<Core::PanoDefinition> panoDef = Core::PanoDefinition::create(panoParser->getRoot());
  ENSURE(panoDef.status());

  // Parse and check the compatible rig
  Potential<Ptv::Parser> rigParser1 = Ptv::Parser::create();
  std::cout << "Parsing rig 1" << std::endl;
  if (!rigParser1->parse("data/rig_pano/rig_1920x1080_good.preset")) {
    std::cerr << rigParser1->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  ENSURE(panoDef->isRigPresetCompatible(&rigParser1->getRoot()));

  // Parse and check the incompatible rig 1 - Incompatible size
  Potential<Ptv::Parser> rigParser2 = Ptv::Parser::create();
  std::cout << "Parsing rig 2" << std::endl;
  if (!rigParser2->parse("data/rig_pano/rig_1280x960_good.preset")) {
    std::cerr << rigParser2->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  ENSURE(!panoDef->isRigPresetCompatible(&rigParser2->getRoot()));

  // Parse and check the incompatible rig 2 - Missing camera
  Potential<Ptv::Parser> rigParser3 = Ptv::Parser::create();
  std::cout << "Parsing rig 3" << std::endl;
  if (!rigParser3->parse("data/rig_pano/rig_wrong_1.preset")) {
    std::cerr << rigParser3->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  ENSURE(!panoDef->isRigPresetCompatible(&rigParser3->getRoot()));

  // Parse and check the incompatible rig 3 - Missing rig camera
  Potential<Ptv::Parser> rigParser4 = Ptv::Parser::create();
  std::cout << "Parsing rig 4" << std::endl;
  if (!rigParser4->parse("data/rig_pano/rig_wrong_2.preset")) {
    std::cerr << rigParser4->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  ENSURE(!panoDef->isRigPresetCompatible(&rigParser4->getRoot()));
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::rigPanoTest();
  return 0;
}
