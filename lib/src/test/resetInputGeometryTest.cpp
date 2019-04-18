// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Testing {

Core::PanoDefinition* getTestPanoDef() {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  Logger::get(Logger::Info) << "parsing" << std::endl;
  if (!parser->parse("data/calibrated_pano_definition.json")) {
    Logger::get(Logger::Error) << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return nullptr;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);

  return panoDef.release();
}

void checkResetGeometry() {
  Logger::get(Logger::Info) << "Creating PanoDefinition" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  Logger::get(Logger::Info) << "Resetting and checking input geometries" << std::endl;
  std::vector<double> HFOVs = {90., 100., 110., 120., 130., 140.};
  for (int i = 0; i < int(panoDef.get()->numVideoInputs()); ++i) {
    Core::InputDefinition& idef = panoDef.get()->getInput(i);
    if (idef.getIsVideoEnabled()) {
      idef.resetGeometries(HFOVs[i]);
      ENSURE_APPROX_EQ(idef.getGeometries().at(0).getEstimatedHorizontalFov(idef), HFOVs[i], .01);
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  // This checks the consistency of the horizontal field of view after an input is reset.
  VideoStitch::Testing::checkResetGeometry();
  return 0;
}
