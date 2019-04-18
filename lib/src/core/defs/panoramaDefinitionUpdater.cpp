// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/panoramaDefinitionUpdater.hpp"

#include "parse/json.hpp"

#include "libvideostitch/mergerMaskUpdater.hpp"
#include "libvideostitch/controlPointListUpdater.hpp"
#include "libvideostitch/inputDefinitionUpdater.hpp"
#include "libvideostitch/overlayInputDefinitionUpdater.hpp"
#include "libvideostitch/rigDef.hpp"

namespace VideoStitch {
namespace Core {

bool PanoramaDefinitionUpdater::validate(std::ostream& os) const { return panoramaDefinition->validate(os); }

bool PanoramaDefinitionUpdater::validateInputMasks() const { return panoramaDefinition->validateInputMasks(); }

const MergerMaskDefinition& PanoramaDefinitionUpdater::getMergerMask() const {
  if (mergerMaskUpdater) {
    return *mergerMaskUpdater;
  }
  return panoramaDefinition->getMergerMask();
}

MergerMaskDefinition& PanoramaDefinitionUpdater::getMergerMask() {
  if (!mergerMaskUpdater) {
    mergerMaskUpdater = std::make_unique<MergerMaskUpdater>(panoramaDefinition->getMergerMask());
    subUpdatersActionLists.push_back(std::ref(mergerMaskUpdater->getActions()));
  }

  actionsToRepeat.push_back(DelayedAction(std::async(
      std::launch::deferred, [this] { mergerMaskUpdater->setToUpdate(updateObjectFuture.get()->getMergerMask()); })));

  return *mergerMaskUpdater;
}

bool PanoramaDefinitionUpdater::getBlendingMaskEnabled() const { return panoramaDefinition->getBlendingMaskEnabled(); }

void PanoramaDefinitionUpdater::setBlendingMaskEnabled(const bool enabled) {
  PRESERVE_ACTION(setBlendingMaskEnabled, panoramaDefinition, enabled);
}

bool PanoramaDefinitionUpdater::getBlendingMaskInterpolationEnabled() const {
  return panoramaDefinition->getBlendingMaskInterpolationEnabled();
}

void PanoramaDefinitionUpdater::setBlendingMaskInterpolationEnabled(const bool enabled) {
  PRESERVE_ACTION(setBlendingMaskInterpolationEnabled, panoramaDefinition, enabled);
}

int64_t PanoramaDefinitionUpdater::getBlendingMaskWidth() const { return panoramaDefinition->getBlendingMaskWidth(); }

int64_t PanoramaDefinitionUpdater::getBlendingMaskHeight() const { return panoramaDefinition->getBlendingMaskHeight(); }

std::vector<size_t> PanoramaDefinitionUpdater::getMasksOrder() const { return panoramaDefinition->getMasksOrder(); }

int PanoramaDefinitionUpdater::getBlendingMaskInputScaleFactor() const {
  return panoramaDefinition->getBlendingMaskInputScaleFactor();
}

std::pair<frameid_t, frameid_t> PanoramaDefinitionUpdater::getBlendingMaskBoundedFrameIds(
    const frameid_t frameId) const {
  return panoramaDefinition->getBlendingMaskBoundedFrameIds(frameId);
}

void PanoramaDefinitionUpdater::removeBlendingMaskFrameIds(const std::vector<frameid_t>& frameIds) {
  return panoramaDefinition->removeBlendingMaskFrameIds(frameIds);
}

std::vector<frameid_t> PanoramaDefinitionUpdater::getBlendingMaskFrameIds() const {
  return panoramaDefinition->getBlendingMaskFrameIds();
}

std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>
PanoramaDefinitionUpdater::getInputIndexPixelDataIfValid(const frameid_t frameId) const {
  return panoramaDefinition->getInputIndexPixelDataIfValid(frameId);
}

const InputDefinition& PanoramaDefinitionUpdater::getInput(readerid_t i) const {
  if (inputsManaged) {
    return *inputUpdaters[i];
  }
  return panoramaDefinition->getInput(i);
}

InputDefinition& PanoramaDefinitionUpdater::getInput(readerid_t i) {
  initInputUpdaters();

  std::weak_ptr<InputDefinitionUpdater> subUpdater = inputUpdaters[i];
  auto futureCopy = updateObjectFuture;
  inputUpdaters[i]->getActions().push_back(DelayedAction(std::async(std::launch::deferred, [i, subUpdater, futureCopy] {
    if (auto sharedSubUpdater = subUpdater.lock()) {
      sharedSubUpdater->setToUpdate(futureCopy.get()->getInput(i));
    }
  })));

  return *inputUpdaters[i];
}

void PanoramaDefinitionUpdater::insertInput(InputDefinition* inputDef, readerid_t i) {
  initInputUpdaters();

  // We take care of releasing inputDef by forwarding it to wrapped object.
  PRESERVE_ACTION_CLONEABLE(insertInput, panoramaDefinition, , , InputDefinition, inputDef, i);

  auto inputDefinitionUpdater = std::make_shared<InputDefinitionUpdater>(inputDef->clone());
  subUpdatersActionLists.push_back(std::ref(inputDefinitionUpdater->getActions()));
  safeInsert(inputUpdaters, inputDefinitionUpdater, i);
}

InputDefinition* PanoramaDefinitionUpdater::popInput(readerid_t i) {
  initInputUpdaters();

  auto result = PanoDefinition::safeRemove(inputUpdaters, i);
  if (!result) {
    return nullptr;
  }

  auto futureCopy = updateObjectFuture;
  actionsToRepeat.push_back(
      DelayedAction(std::async(std::launch::deferred, [futureCopy, i]() { futureCopy.get()->removeInput(i); })));

  auto removeIter =
      std::find_if(std::begin(subUpdatersActionLists), std::end(subUpdatersActionLists),
                   [&result](std::vector<DelayedAction>& value) { return &(result->getActions()) == &value; });
  if (removeIter != std::end(subUpdatersActionLists)) {
    subUpdatersActionLists.erase(removeIter);
  }
  return result->clone();
}

bool PanoramaDefinitionUpdater::removeInput(int i) {
  // calling base class function
  // the call to popInput(int i) inside base class function
  // will be redirected to this class's function
  return PanoDefinition::removeInput(i);
}

readerid_t PanoramaDefinitionUpdater::numInputs() const {
  if (inputsManaged) {
    return (readerid_t)inputUpdaters.size();
  }
  return panoramaDefinition->numInputs();
}

const OverlayInputDefinition& PanoramaDefinitionUpdater::getOverlay(overlayreaderid_t i) const {
  if (overlaysManaged) {
    return *overlayUpdaters[i];
  }
  return panoramaDefinition->getOverlay(i);
}

OverlayInputDefinition& PanoramaDefinitionUpdater::getOverlay(overlayreaderid_t i) {
  initOverlayUpdaters();

  std::weak_ptr<OverlayInputDefinitionUpdater> subUpdater = overlayUpdaters[i];
  auto futureCopy = updateObjectFuture;
  overlayUpdaters[i]->getActions().push_back(
      DelayedAction(std::async(std::launch::deferred, [i, subUpdater, futureCopy] {
        if (auto sharedSubUpdater = subUpdater.lock()) {
          sharedSubUpdater->setToUpdate(futureCopy.get()->getOverlay(i));
        }
      })));

  return *overlayUpdaters[i];
}

void PanoramaDefinitionUpdater::insertOverlay(OverlayInputDefinition* overlayDef, overlayreaderid_t i) {
  initOverlayUpdaters();

  // We take care of releasing inputDef by forwarding it to wrapped object.
  PRESERVE_ACTION_CLONEABLE(insertOverlay, panoramaDefinition, , , OverlayInputDefinition, overlayDef, i);

  auto overlayInputDefinitionUpdater = std::make_shared<OverlayInputDefinitionUpdater>(overlayDef->clone());
  subUpdatersActionLists.push_back(std::ref(overlayInputDefinitionUpdater->getActions()));
  safeInsert(overlayUpdaters, overlayInputDefinitionUpdater, i);
}

OverlayInputDefinition* PanoramaDefinitionUpdater::popOverlay(overlayreaderid_t i) {
  initOverlayUpdaters();

  auto result = PanoDefinition::safeRemove(overlayUpdaters, i);
  if (!result) {
    return nullptr;
  }

  auto futureCopy = updateObjectFuture;
  actionsToRepeat.push_back(
      DelayedAction(std::async(std::launch::deferred, [futureCopy, i]() { futureCopy.get()->removeOverlay(i); })));

  auto removeIter =
      std::find_if(std::begin(subUpdatersActionLists), std::end(subUpdatersActionLists),
                   [&result](std::vector<DelayedAction>& value) { return &(result->getActions()) == &value; });
  if (removeIter != std::end(subUpdatersActionLists)) {
    subUpdatersActionLists.erase(removeIter);
  }
  return result->clone();
}

bool PanoramaDefinitionUpdater::removeOverlay(overlayreaderid_t i) {
  // calling base class function
  // the call to popOverlay(int i) inside base class function
  // will be redirected to this class's function
  return PanoDefinition::removeOverlay(i);
}

overlayreaderid_t PanoramaDefinitionUpdater::numOverlays() const {
  if (overlaysManaged) {
    return (overlayreaderid_t)overlayUpdaters.size();
  }
  return panoramaDefinition->numOverlays();
}

const ControlPointListDefinition& PanoramaDefinitionUpdater::getControlPointListDef() const {
  if (controlPointsListUpdater) {
    return *controlPointsListUpdater;
  }

  return panoramaDefinition->getControlPointListDef();
}

ControlPointListDefinition& PanoramaDefinitionUpdater::getControlPointListDef() {
  // Todo: more macros?
  if (!controlPointsListUpdater) {
    controlPointsListUpdater = std::make_unique<ControlPointsListUpdater>(panoramaDefinition->getControlPointListDef());
    subUpdatersActionLists.push_back(std::ref(controlPointsListUpdater->getActions()));
  }

  actionsToRepeat.push_back(DelayedAction(std::async(std::launch::deferred, [this] {
    controlPointsListUpdater->setToUpdate(updateObjectFuture.get()->getControlPointListDef());
  })));

  return *controlPointsListUpdater;
}

bool PanoramaDefinitionUpdater::getPrecomputedCoordinateBuffer() const {
  return panoramaDefinition->getPrecomputedCoordinateBuffer();
}

void PanoramaDefinitionUpdater::setPrecomputedCoordinateBuffer(const bool b) {
  PRESERVE_ACTION(setPrecomputedCoordinateBuffer, panoramaDefinition, b);
}

double PanoramaDefinitionUpdater::getPrecomputedCoordinateShrinkFactor() const {
  return panoramaDefinition->getPrecomputedCoordinateShrinkFactor();
}

void PanoramaDefinitionUpdater::setPrecomputedCoordinateShrinkFactor(const double b) {
  PRESERVE_ACTION(setPrecomputedCoordinateShrinkFactor, panoramaDefinition, b);
}

int64_t PanoramaDefinitionUpdater::getWidth() const { return panoramaDefinition->getWidth(); }

int64_t PanoramaDefinitionUpdater::getHeight() const { return panoramaDefinition->getHeight(); }

int64_t PanoramaDefinitionUpdater::getLength() const { return panoramaDefinition->getLength(); }

const Ptv::Value* PanoramaDefinitionUpdater::getPostprocessors() const {
  return panoramaDefinition->getPostprocessors();
}

double PanoramaDefinitionUpdater::getHFOV() const { return panoramaDefinition->getHFOV(); }

double PanoramaDefinitionUpdater::getVFOV() const { return panoramaDefinition->getVFOV(); }

double PanoramaDefinitionUpdater::getSphereScale() const { return panoramaDefinition->getSphereScale(); }

void PanoramaDefinitionUpdater::setSphereScale(double scale) {
  PRESERVE_ACTION(setSphereScale, panoramaDefinition, scale);
}

void PanoramaDefinitionUpdater::setCalibrationCost(double cost) {
  PRESERVE_ACTION(setCalibrationCost, panoramaDefinition, cost);
}

double PanoramaDefinitionUpdater::getCalibrationCost() const { return panoramaDefinition->getCalibrationCost(); }

void PanoramaDefinitionUpdater::setCalibrationInitialHFOV(double hfov) {
  PRESERVE_ACTION(setCalibrationInitialHFOV, panoramaDefinition, hfov);
}

double PanoramaDefinitionUpdater::getCalibrationInitialHFOV() const {
  return panoramaDefinition->getCalibrationInitialHFOV();
}

void PanoramaDefinitionUpdater::setHasBeenCalibrationDeshuffled(const bool deshuffled) {
  PRESERVE_ACTION(setHasBeenCalibrationDeshuffled, panoramaDefinition, deshuffled);
}

bool PanoramaDefinitionUpdater::hasBeenCalibrationDeshufled() const {
  return panoramaDefinition->hasBeenCalibrationDeshufled();
}

void PanoramaDefinitionUpdater::setCalibrationControlPointList(const ControlPointList& list) {
  PRESERVE_ACTION(setCalibrationControlPointList, panoramaDefinition, list);
}

const ControlPointList& PanoramaDefinitionUpdater::getCalibrationControlPointList() const {
  return panoramaDefinition->getCalibrationControlPointList();
}

void PanoramaDefinitionUpdater::setCalibrationRigPresets(RigDefinition* rigDef) {
  PRESERVE_ACTION_CLONEABLE(setCalibrationRigPresets, panoramaDefinition, , , RigDefinition, rigDef);
}

const RigDefinition& PanoramaDefinitionUpdater::getCalibrationRigPresets() const {
  return panoramaDefinition->getCalibrationRigPresets();
}

std::string PanoramaDefinitionUpdater::getCalibrationRigPresetsName() const {
  return panoramaDefinition->getCalibrationRigPresetsName();
}

bool PanoramaDefinitionUpdater::hasCalibrationRigPresets() const {
  return panoramaDefinition->hasCalibrationRigPresets();
}

void PanoramaDefinitionUpdater::setWidth(uint64_t w) { PRESERVE_ACTION(setWidth, panoramaDefinition, w); }

void PanoramaDefinitionUpdater::setHeight(uint64_t h) { PRESERVE_ACTION(setHeight, panoramaDefinition, h); }

void PanoramaDefinitionUpdater::setLength(uint64_t l) { PRESERVE_ACTION(setLength, panoramaDefinition, l); }

const CurveTemplate<double>& PanoramaDefinitionUpdater::getRedCB() const { return panoramaDefinition->getRedCB(); }

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceRedCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceRedCB, panoramaDefinition, delete, return, double, newCurve);
}

void PanoramaDefinitionUpdater::resetRedCB() { PRESERVE_ACTION(resetRedCB, panoramaDefinition); }

void PanoramaDefinitionUpdater::replaceRedCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceRedCB, panoramaDefinition, , , double, newCurve);
}

void PanoramaDefinitionUpdater::resetGreenCB() { PRESERVE_ACTION(resetGreenCB, panoramaDefinition); }

const CurveTemplate<double>& PanoramaDefinitionUpdater::getGreenCB() const { return panoramaDefinition->getGreenCB(); }

void PanoramaDefinitionUpdater::replaceGreenCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceGreenCB, panoramaDefinition, , , double, newCurve);
}

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceGreenCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceGreenCB, panoramaDefinition, delete, return, double, newCurve);
}

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceBlueCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceBlueCB, panoramaDefinition, delete, return, double, newCurve);
}

void PanoramaDefinitionUpdater::resetBlueCB() { PRESERVE_ACTION(resetBlueCB, panoramaDefinition); }

const CurveTemplate<double>& PanoramaDefinitionUpdater::getBlueCB() const { return panoramaDefinition->getBlueCB(); }

void PanoramaDefinitionUpdater::replaceBlueCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceBlueCB, panoramaDefinition, , , double, newCurve);
}

const CurveTemplate<double>& PanoramaDefinitionUpdater::getExposureValue() const {
  return panoramaDefinition->getExposureValue();
}

void PanoramaDefinitionUpdater::resetExposureValue() { PRESERVE_ACTION(resetExposureValue, panoramaDefinition); }

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceExposureValue(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceExposureValue, panoramaDefinition, delete, return, double, newCurve);
}

void PanoramaDefinitionUpdater::replaceExposureValue(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceExposureValue, panoramaDefinition, , , double, newCurve);
}

void PanoramaDefinitionUpdater::setProjection(PanoProjection format) {
  PRESERVE_ACTION(setProjection, panoramaDefinition, format);
}

void PanoramaDefinitionUpdater::setHFOV(double hFov) { PRESERVE_ACTION(setHFOV, panoramaDefinition, hFov); }

void PanoramaDefinitionUpdater::setVFOV(double vFov) { PRESERVE_ACTION(setVFOV, panoramaDefinition, vFov); }

void PanoramaDefinitionUpdater::resetGlobalOrientation() {
  PRESERVE_ACTION(resetGlobalOrientation, panoramaDefinition);
}

const CurveTemplate<Quaternion<double>>& PanoramaDefinitionUpdater::getGlobalOrientation() const {
  return panoramaDefinition->getGlobalOrientation();
}

CurveTemplate<Quaternion<double>>* PanoramaDefinitionUpdater::displaceGlobalOrientation(
    CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceGlobalOrientation, panoramaDefinition, delete, return, Quaternion<double>, newCurve);
}

void PanoramaDefinitionUpdater::replaceGlobalOrientation(CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceGlobalOrientation, panoramaDefinition, , , Quaternion<double>, newCurve);
}

CurveTemplate<Quaternion<double>>* PanoramaDefinitionUpdater::displaceStabilization(
    CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceStabilization, panoramaDefinition, delete, return, Quaternion<double>, newCurve);
}

void PanoramaDefinitionUpdater::resetStabilization() { PRESERVE_ACTION(resetStabilization, panoramaDefinition); }

const CurveTemplate<Quaternion<double>>& PanoramaDefinitionUpdater::getStabilization() const {
  return panoramaDefinition->getStabilization();
}

void PanoramaDefinitionUpdater::replaceStabilization(CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceStabilization, panoramaDefinition, , , Quaternion<double>, newCurve);
}

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceStabilizationYaw(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceStabilizationYaw, panoramaDefinition, delete, return, double, newCurve);
}

void PanoramaDefinitionUpdater::replaceStabilizationYaw(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceStabilizationYaw, panoramaDefinition, , , double, newCurve);
}

const CurveTemplate<double>& PanoramaDefinitionUpdater::getStabilizationYaw() const {
  return panoramaDefinition->getStabilizationYaw();
}

void PanoramaDefinitionUpdater::resetStabilizationYaw() { PRESERVE_ACTION(resetStabilizationYaw, panoramaDefinition); }

void PanoramaDefinitionUpdater::replaceStabilizationPitch(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceStabilizationPitch, panoramaDefinition, , , double, newCurve);
}

void PanoramaDefinitionUpdater::resetStabilizationPitch() {
  PRESERVE_ACTION(resetStabilizationPitch, panoramaDefinition);
}

const CurveTemplate<double>& PanoramaDefinitionUpdater::getStabilizationPitch() const {
  return panoramaDefinition->getStabilizationPitch();
}

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceStabilizationPitch(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceStabilizationPitch, panoramaDefinition, delete, return, double, newCurve);
}

const CurveTemplate<double>& PanoramaDefinitionUpdater::getStabilizationRoll() const {
  return panoramaDefinition->getStabilizationRoll();
}

void PanoramaDefinitionUpdater::resetStabilizationRoll() {
  PRESERVE_ACTION(resetStabilizationRoll, panoramaDefinition);
}

CurveTemplate<double>* PanoramaDefinitionUpdater::displaceStabilizationRoll(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceStabilizationRoll, panoramaDefinition, delete, return, double, newCurve);
}

void PanoramaDefinitionUpdater::replaceStabilizationRoll(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceStabilizationRoll, panoramaDefinition, , , double, newCurve);
}

void PanoramaDefinitionUpdater::computeOptimalPanoSize(unsigned& width, unsigned& height) const {
  panoramaDefinition->computeOptimalPanoSize(width, height);
}

PanoramaDefinitionUpdater::~PanoramaDefinitionUpdater() {}

void PanoramaDefinitionUpdater::apply(PanoDefinition& updateValue) {
  DeferredUpdater::apply(updateValue);
  panoramaDefinition = std::unique_ptr<PanoDefinition>(updateValue.clone());
}

void PanoramaDefinitionUpdater::initInputUpdaters() {
  if (inputsManaged) {
    return;
  }
  for (auto i = 0; i < int(numInputs()); i++) {
    auto inputDefinitionUpdater = std::make_shared<InputDefinitionUpdater>(panoramaDefinition->getInput(i));
    subUpdatersActionLists.push_back(std::ref(inputDefinitionUpdater->getActions()));
    inputUpdaters.push_back(inputDefinitionUpdater);
  }
  inputsManaged = true;
}

void PanoramaDefinitionUpdater::initOverlayUpdaters() {
  if (overlaysManaged) {
    return;
  }
  for (auto i = 0; i < int(numOverlays()); i++) {
    auto overlayInputDefinitionUpdater =
        std::make_shared<OverlayInputDefinitionUpdater>(panoramaDefinition->getOverlay(i));
    subUpdatersActionLists.push_back(std::ref(overlayInputDefinitionUpdater->getActions()));
    overlayUpdaters.push_back(overlayInputDefinitionUpdater);
  }
  overlaysManaged = true;
}

PanoramaDefinitionUpdater::PanoramaDefinitionUpdater(const PanoDefinition& panoDefinition)
    : panoramaDefinition(std::unique_ptr<PanoDefinition>(panoDefinition.clone())) {}

PanoramaDefinitionUpdater::PanoramaDefinitionUpdater(PanoramaDefinitionUpdater&& rhs)
    : PanoDefinition(std::move(rhs)),
      DeferredUpdater(std::move(rhs)),
      panoramaDefinition(std::move(rhs.panoramaDefinition)) {}

}  // namespace Core
}  // namespace VideoStitch
