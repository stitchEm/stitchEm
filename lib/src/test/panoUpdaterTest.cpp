// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoramaDefinitionUpdater.hpp"
#include "libvideostitch/mergerMaskUpdater.hpp"
#include "common/ptv.hpp"

#include <memory>
#include <string>
#include <map>

/** Todo:
 * test set, apply
 * test set subentities, apply
 * test attempt several apply (not supported, but should not crash)
 * test add/remove entities thing on panodefinition
 * test clone
 * test serialize (after sub update)
 * test no override happens
 */

using namespace VideoStitch::Core;

namespace VideoStitch {
namespace Testing {

static const std::string DEFAULT_PANO_FILENAME = "data/4i_default_pano_definition.json";
static const std::string OVERLAY_PANO_FILENAME = "data/4i_overlay_pano_definition.json";

std::shared_ptr<PanoDefinition> getDefaultPano(const std::string& filename) {
  auto parser = Ptv::Parser::create();

  if (!parser->parse(filename)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  return std::shared_ptr<PanoDefinition>(PanoDefinition::create(parser->getRoot()));
}

void testSetApply() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->setCalibrationCost(1001001);
  tUpdater.setCalibrationCost(1001001);
  ENSURE_EQ(tUpdater.getCalibrationCost(), 1001001.);

  tPano->getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  tPano->resetBlueCB();
  tUpdater.resetBlueCB();

  tPano->setProjection(PanoProjection(PanoProjection::Type::Stereographic));
  tUpdater.setProjection(PanoProjection(PanoProjection::Type::Stereographic));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testSetApplyMergerMask() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getMergerMask().setHeight(1234);
  tUpdater.getMergerMask().setHeight(1234);
  ENSURE_EQ(tUpdater.getMergerMask().getHeight(), int64_t(1234));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testSetApplyInputDef() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  delete tPano->getInput(1).displaceBlueCB(new CurveTemplate<double>(1234));
  delete tUpdater.getInput(1).displaceBlueCB(new CurveTemplate<double>(1234));
  ENSURE_EQ(tUpdater.getInput(1).getBlueCB().getConstantValue(), 1234.);

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);

  // ensure that clone operates correctly
  auto panoUPDClone = std::shared_ptr<PanoDefinition>(tUpdater.clone());
  ENSURE(*tPano == *panoUPDClone);
}

void testSerialize() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  delete tPano->getInput(1).displaceBlueCB(new CurveTemplate<double>(1234));
  delete tUpdater.getInput(1).displaceBlueCB(new CurveTemplate<double>(1234));

  delete tPano->getOverlay(1).displaceAlphaCurve(new CurveTemplate<double>(0.5));
  delete tUpdater.getOverlay(1).displaceAlphaCurve(new CurveTemplate<double>(0.5));

  tPano->getMergerMask().setHeight(1234);
  tUpdater.getMergerMask().setHeight(1234);

  tPano->setCalibrationCost(1001001);
  tUpdater.setCalibrationCost(1001001);

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);

  auto serializedPano = std::unique_ptr<Ptv::Value>(tPano->serialize());
  auto serializedPanoUpd = std::unique_ptr<Ptv::Value>(tUpdater.serialize());

  ENSURE_EQ(serializedPano->getJsonStr(), serializedPanoUpd->getJsonStr());

  // ensure that clone operates correctly
  auto panoUPDClone = std::shared_ptr<PanoDefinition>(tUpdater.clone());
  ENSURE(*tPano == *panoUPDClone);
}

void testDivergeChanges() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));

  tUpdater.getMergerMask().setHeight(1234);
  tUpdater.getOverlay(1).setGlobalOrientationApplied(true);
  tUpdater.getOverlay(1).setHeight(1234);

  tPano->setCalibrationCost(1001001);
  tUpdater.setCalibrationCost(1001001);

  auto panoCloneRecent = std::shared_ptr<PanoDefinition>(tPano->clone());
  tUpdater.apply(*panoCloneRecent);

  tPano->getMergerMask().setHeight(1234);
  tPano->getOverlay(1).setGlobalOrientationApplied(true);
  tPano->getOverlay(1).setHeight(1234);
  ENSURE(*tPano == *panoCloneRecent);
}

void testRemoveInputDef() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));

  tPano->removeInput(1);
  tUpdater.removeInput(1);

  tPano->getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  ENSURE_EQ(tUpdater.getInput(1).getBlueCB().getConstantValue(), 1234.);

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testRemoveInputDefNoGet() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->removeInput(1);
  tUpdater.removeInput(1);

  tPano->getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(1).replaceBlueCB(new CurveTemplate<double>(1234));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testRemoveAddModify() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getInput(3).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(3).replaceBlueCB(new CurveTemplate<double>(1234));

  auto input1tp = tPano->popInput(1);
  auto input1tu = tUpdater.popInput(1);

  tPano->getInput(0).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(0).replaceBlueCB(new CurveTemplate<double>(1234));

  tPano->insertInput(input1tp, 3);
  tUpdater.insertInput(input1tu, 3);

  tPano->getInput(3).replaceBlueCB(new CurveTemplate<double>(1234));
  tUpdater.getInput(3).replaceBlueCB(new CurveTemplate<double>(1234));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testDataPreserve() {
  auto tPano = getDefaultPano(DEFAULT_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  std::map<videoreaderid_t, std::string> buffer;
  buffer.insert({0, "123"});
  buffer.insert({1, "123"});

  tPano->getMergerMask().setInputIndexPixelData(buffer, 4, 1, 1);

  tUpdater.getMergerMask().setInputIndexPixelData(buffer, 4, 1, 1);

  auto serializedUpdater = std::unique_ptr<Ptv::Value>(tUpdater.serialize());
  auto serializedPano = std::unique_ptr<Ptv::Value>(tPano->serialize());

  ENSURE_EQ(serializedPano->getJsonStr(), serializedUpdater->getJsonStr());
  auto serializedPanoClone = std::unique_ptr<Ptv::Value>(panoClone->serialize());
  ENSURE_NEQ(serializedPanoClone->getJsonStr(), serializedPano->getJsonStr());

  tUpdater.apply(*panoClone);

  serializedPanoClone = std::unique_ptr<Ptv::Value>(panoClone->serialize());
  ENSURE_EQ(serializedPanoClone->getJsonStr(), serializedPano->getJsonStr());
}

void testSetApplyOverlayDef() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  delete tPano->getOverlay(1).displaceTransXCurve(new CurveTemplate<double>(0.1));
  delete tUpdater.getOverlay(1).displaceTransXCurve(new CurveTemplate<double>(0.1));
  ENSURE_EQ(tUpdater.getOverlay(1).getTransXCurve().getConstantValue(), 0.1);

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);

  // ensure that clone operates correctly
  auto panoUPDClone = std::shared_ptr<PanoDefinition>(tUpdater.clone());
  ENSURE(*tPano == *panoUPDClone);
}

void testRemoveOverlayDef() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));

  tPano->removeOverlay(1);
  tUpdater.removeOverlay(1);

  tPano->getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));
  ENSURE_EQ(tUpdater.getOverlay(1).getTransZCurve().getConstantValue(), 0.1);

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testRemoveOverlayDefNoGet() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->removeOverlay(1);
  tUpdater.removeOverlay(1);

  tPano->getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(1).replaceTransZCurve(new CurveTemplate<double>(0.1));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testRemoveOverlayAddModify() {
  auto tPano = getDefaultPano(OVERLAY_PANO_FILENAME);
  auto panoClone = std::shared_ptr<PanoDefinition>(tPano->clone());

  PanoramaDefinitionUpdater tUpdater(*tPano);

  tPano->getOverlay(3).replaceTransYCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(3).replaceTransYCurve(new CurveTemplate<double>(0.1));

  auto overlay1tp = tPano->popOverlay(1);
  auto overlay1tu = tUpdater.popOverlay(1);

  tPano->getOverlay(0).replaceTransYCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(0).replaceTransYCurve(new CurveTemplate<double>(0.1));

  tPano->insertOverlay(overlay1tp, 3);
  tUpdater.insertOverlay(overlay1tu, 3);

  tPano->getOverlay(3).replaceTransYCurve(new CurveTemplate<double>(0.1));
  tUpdater.getOverlay(3).replaceTransYCurve(new CurveTemplate<double>(0.1));

  ENSURE(!(*tPano == *panoClone));
  tUpdater.apply(*panoClone);

  ENSURE(*tPano == *panoClone);
}

void testMaxNumberInputs() {
  // VSA-7378
  auto pano = getDefaultPano(DEFAULT_PANO_FILENAME);
  int initialNumInputs = pano->numVideoInputs();

  const std::string inputConfig =
      "{"
      " \"width\": 20,"
      " \"height\": 10,"
      " \"response\": \"gamma\","
      " \"audio_enabled\": true,"
      " \"reader_config\": {},"       // Dummy
      " \"proj\": \"rectilinear\"}";  // Dummy
  const std::unique_ptr<Ptv::Value> inputDefPtv(makePtvValue(inputConfig));

  ENSURE(pano->numAudioInputs() > 0);
  for (int i = initialNumInputs; i < MAX_VIDEO_INPUTS; ++i) {
    pano->insertInput(Core::InputDefinition::create(*inputDefPtv), -1);
  }
  ENSURE(pano->validate(std::cerr));

  // test when we create MAX_VIDEO_INPUTS +1
  pano->insertInput(Core::InputDefinition::create(*inputDefPtv), -1);
  ENSURE(!pano->validate(std::cerr));
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testSetApply();
  VideoStitch::Testing::testSetApplyMergerMask();
  VideoStitch::Testing::testSetApplyInputDef();
  VideoStitch::Testing::testSerialize();
  VideoStitch::Testing::testDivergeChanges();
  VideoStitch::Testing::testRemoveInputDef();
  VideoStitch::Testing::testRemoveInputDefNoGet();
  VideoStitch::Testing::testRemoveAddModify();
  VideoStitch::Testing::testDataPreserve();
  VideoStitch::Testing::testSetApplyOverlayDef();
  VideoStitch::Testing::testRemoveOverlayDef();
  VideoStitch::Testing::testRemoveOverlayDefNoGet();
  VideoStitch::Testing::testRemoveOverlayAddModify();
  VideoStitch::Testing::testMaxNumberInputs();

  return 0;
}
