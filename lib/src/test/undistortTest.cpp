// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "common/fakeReader.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/undistortController.hpp"
#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/overrideDef.hpp"

#include <fstream>

namespace VideoStitch {
namespace Testing {

// #define DEBUG_OUTPUT_UPDATED_PANO_DEF

// check failure modes

Core::PanoDefinition* getPanoDef(const std::string& filename) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  Logger::get(Logger::Info) << "parsing" << std::endl;
  if (!parser->parse(filename)) {
    Logger::get(Logger::Error) << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return nullptr;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);

  return panoDef.release();
}

void createUpdatedPtvTest(const Core::OverrideOutputDefinition& outputDef, std::string comparisonFileName) {
  Logger::get(Logger::Info) << "Creating PanoDefinition" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getPanoDef("data/AQ1610012338_pano.ptv"));
  Logger::get(Logger::Info) << "Attempting to create pano def without undistortion" << std::endl;

  FakeReaderFactory* fakeReaderFactory = new FakeReaderFactory(0);

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(Core::AudioPipeDefinition::createDefault());
  ENSURE(audioPipeDef != nullptr);

  Potential<Core::UndistortController> ctrl =
      Core::createUndistortController(*panoDef, *audioPipeDef, fakeReaderFactory, outputDef);
  ENSURE(ctrl.ok());

  Potential<Core::PanoDefinition> updatedPano = ctrl->createPanoDefWithoutDistortion();
  ENSURE(updatedPano.ok());

#ifdef DEBUG_OUTPUT_UPDATED_PANO_DEF
  std::ofstream ofs("/tmp/" + comparisonFileName, std::ios_base::out);
  ENSURE(ofs.is_open());
  VideoStitch::Ptv::Value* root = updatedPano->serialize();
  root->printJson(ofs);
  delete root;
#endif

  ENSURE(*panoDef == *panoDef, "Sanity check");
  ENSURE(!(*updatedPano.object() == *panoDef), "Undistortion should have updated pano");

  std::unique_ptr<Core::PanoDefinition> expected(getPanoDef("data/" + comparisonFileName));
  ENSURE(*expected == *updatedPano.object(),
         "Newly created undistorted pano should equal expected .ptv loaded from disk");
}

void createUpdatedPtvTest() {
  {
    Core::OverrideOutputDefinition outputDef{};
    createUpdatedPtvTest(outputDef, "AQ1610012338_pano_undistorted.ptv");
  }

  {
    Core::OverrideOutputDefinition outputDef{};
    outputDef.manualFocal = true;
    outputDef.overrideFocal = 23;
    createUpdatedPtvTest(outputDef, "AQ1610012338_pano_undistorted_focal.ptv");
  }

  {
    Core::OverrideOutputDefinition outputDef{};
    outputDef.resetRotation = true;
    createUpdatedPtvTest(outputDef, "AQ1610012338_pano_undistorted_reset_rotation.ptv");
  }

  {
    Core::OverrideOutputDefinition outputDef{};
    outputDef.changeOutputFormat = true;
    outputDef.newFormat = Core::InputDefinition::Format::FullFrameFisheye;
    createUpdatedPtvTest(outputDef, "AQ1610012338_pano_undistorted_fisheye.ptv");
  }
}

void checkResetDistortion() {
  Logger::get(Logger::Info) << "Creating PanoDefinition" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getPanoDef("data/AQ1610012338_pano.ptv"));
  Logger::get(Logger::Info) << "Resetting and checking distortion" << std::endl;
  for (Core::InputDefinition& idef : panoDef->getVideoInputs()) {
    ENSURE_EQ(idef.computeFocalWithoutDistortion(), 720.);

    Core::GeometryDefinition origGeometry = idef.getGeometries().at(0);
    idef.resetDistortion();

    // these values should have been reset
    ENSURE_EQ(idef.getGeometries().at(0).getDistortA(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortB(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortC(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getCenterX(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getCenterY(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortP1(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortP2(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortS1(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortS2(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortS3(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortS4(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortTau1(), 0.);
    ENSURE_EQ(idef.getGeometries().at(0).getDistortTau2(), 0.);

    // these values should have been kept
    ENSURE_EQ(idef.getGeometries().at(0).getYaw(), origGeometry.getYaw());
    ENSURE_EQ(idef.getGeometries().at(0).getPitch(), origGeometry.getPitch());
    ENSURE_EQ(idef.getGeometries().at(0).getRoll(), origGeometry.getRoll());

    ENSURE_EQ(idef.getGeometries().at(0).getTranslationX(), origGeometry.getTranslationX());
    ENSURE_EQ(idef.getGeometries().at(0).getTranslationY(), origGeometry.getTranslationY());
    ENSURE_EQ(idef.getGeometries().at(0).getTranslationZ(), origGeometry.getTranslationZ());

    ENSURE_EQ(idef.getGeometries().at(0).getHorizontalFocal(), origGeometry.getHorizontalFocal());
    ENSURE_EQ(idef.getGeometries().at(0).getVerticalFocal(), origGeometry.getVerticalFocal());
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::checkResetDistortion();

  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::createUpdatedPtvTest();
  return 0;
}
