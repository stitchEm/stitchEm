// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "common/angles.hpp"
#include "common/container.hpp"
#include "parse/json.hpp"
#include "core/geoTransform.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/curves.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/ptv.hpp"

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <set>
#include <iterator>
#include <sstream>

namespace VideoStitch {
namespace Core {

/**
  Note:
  1) Don't forget to update *definitionUpdater, when you update definition.
  2) In algorithms inside definition if you work with some subcomponents (for this one: MergerMaskDefinition,
  ControlPointsListDefinition, InputDefinitions) Make sure to use virtual getter methods in order to access them and NOT
  direct access through the pimpl.
*/
PanoDefinition::Pimpl::Pimpl()
    : mergerMask(nullptr),
      controlPointList(nullptr),
      rigDefinition(nullptr),
      postprocessors(nullptr),
      width(0),
      height(0),
      cropLeft(std::numeric_limits<int64_t>::max()),
      cropRight(std::numeric_limits<int64_t>::max()),
      cropTop(std::numeric_limits<int64_t>::max()),
      cropBottom(std::numeric_limits<int64_t>::max()),
      hFOV(0.0f),
      exposureValue(new Curve(PTV_DEFAULT_PANODEF_EXPOSURE)),
      redCB(new Curve(PTV_DEFAULT_PANODEF_REDCB)),
      greenCB(new Curve(PTV_DEFAULT_PANODEF_GREENCB)),
      blueCB(new Curve(PTV_DEFAULT_PANODEF_BLUECB)),
      wrap(1),
      projection(PanoProjection::Equirectangular),
      orientationCurve(new QuaternionCurve(Quaternion<double>())),
      stabilizationCurve(new QuaternionCurve(Quaternion<double>())),
      stabilizationYawCurve(new Curve(PTV_DEFAULT_PANODEF_STAB_YAW)),
      stabilizationPitchCurve(new Curve(PTV_DEFAULT_PANODEF_STAB_PITCH)),
      stabilizationRollCurve(new Curve(PTV_DEFAULT_PANODEF_STAB_ROLL)),
      sphereScale(PTV_DEFAULT_PANODEF_SPHERE_SCALE),
      calibrationCost(-1.0),
      calibrationInitialHFOV(0.),
      calibrationDeshuffled(false),
      precomputedCoodinateBuffer(false),
      precomputedCoodinateShrinkFactor(1.0f),
      inputsMapInterpolationEnabled(true) {}

PanoDefinition::PanoDefinition() : pimpl(new Pimpl()) {}

PanoDefinition::PanoDefinition(PanoDefinition&& rhs) : pimpl(rhs.pimpl) { rhs.pimpl = nullptr; }

PanoDefinition* PanoDefinition::clone() const {
  PanoDefinition* result = new PanoDefinition();

#define AUTO_FIELD_COPY(field) result->set##field(get##field())
#define AUTO_CURVE_COPY(curve) result->replace##curve(get##curve().clone())
  AUTO_FIELD_COPY(Width);
  AUTO_FIELD_COPY(Height);
  AUTO_FIELD_COPY(Length);
  AUTO_FIELD_COPY(HFOV);
  AUTO_FIELD_COPY(PrecomputedCoordinateBuffer);
  AUTO_FIELD_COPY(PrecomputedCoordinateShrinkFactor);
  AUTO_CURVE_COPY(ExposureValue);
  AUTO_CURVE_COPY(RedCB);
  AUTO_CURVE_COPY(GreenCB);
  AUTO_CURVE_COPY(BlueCB);
  AUTO_FIELD_COPY(Projection);
  AUTO_FIELD_COPY(SphereScale);
  AUTO_FIELD_COPY(CalibrationCost);
  AUTO_CURVE_COPY(GlobalOrientation);
  AUTO_CURVE_COPY(Stabilization);
  AUTO_CURVE_COPY(StabilizationYaw);
  AUTO_CURVE_COPY(StabilizationPitch);
  AUTO_CURVE_COPY(StabilizationRoll);
#undef AUTO_FIELD_COPY
#undef AUTO_CURVE_COPY

  result->setHasBeenCalibrationDeshuffled(hasBeenCalibrationDeshufled());

  for (readerid_t i = 0; i < numInputs(); ++i) {
    result->pimpl->inputs.push_back(getInput(i).clone());
  }

  for (overlayreaderid_t i = 0; i < numOverlays(); ++i) {
    result->pimpl->overlays.push_back(getOverlay(i).clone());
  }

  result->pimpl->mergerMask = getMergerMask().clone();
  result->pimpl->controlPointList = getControlPointListDef().clone();
  result->pimpl->rigDefinition = getCalibrationRigPresets().clone();

  delete result->pimpl->postprocessors;
  result->pimpl->postprocessors = getPostprocessors() ? getPostprocessors()->clone() : nullptr;
  return result;
}

bool PanoDefinition::operator==(const PanoDefinition& other) const {
#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getWidth) && FIELD_EQUAL(getHeight) && FIELD_EQUAL(getLength) && FIELD_EQUAL(getHFOV) &&
        FIELD_EQUAL(getExposureValue) && FIELD_EQUAL(getRedCB) && FIELD_EQUAL(getGreenCB) && FIELD_EQUAL(getBlueCB) &&
        FIELD_EQUAL(getControlPointListDef) && FIELD_EQUAL(getCalibrationRigPresets) &&
        FIELD_EQUAL(getCalibrationCost) && FIELD_EQUAL(getCalibrationInitialHFOV) &&
        FIELD_EQUAL(hasBeenCalibrationDeshufled) && FIELD_EQUAL(getPrecomputedCoordinateBuffer) &&
        FIELD_EQUAL(getPrecomputedCoordinateShrinkFactor) && FIELD_EQUAL(getProjection) && FIELD_EQUAL(numInputs) &&
        FIELD_EQUAL(getBlendingMaskWidth) && FIELD_EQUAL(getBlendingMaskHeight) &&
        FIELD_EQUAL(getBlendingMaskEnabled) && FIELD_EQUAL(getGlobalOrientation) && FIELD_EQUAL(getSphereScale) &&
        FIELD_EQUAL(getStabilization))) {
    return false;
  }
  for (readerid_t i = 0; i < numInputs(); ++i) {
    if (!(getInput(i) == other.getInput(i))) {
      return false;
    }
  }
  for (overlayreaderid_t i = 0; i < numOverlays(); ++i) {
    if (!(getOverlay(i) == other.getOverlay(i))) {
      return false;
    }
  }
  if (getPostprocessors() != NULL && other.getPostprocessors() != NULL) {
    if (!(*(getPostprocessors()) == *other.getPostprocessors())) {
      return false;
    }
  } else if (getPostprocessors() != NULL || other.getPostprocessors() != NULL) {
    return false;
  }
  std::vector<frameid_t> frameIds = getBlendingMaskFrameIds();
  std::vector<frameid_t> frameIdsOther = other.getBlendingMaskFrameIds();
  if (frameIds.size() != frameIdsOther.size()) {
    return false;
  }
  std::sort(frameIds.begin(), frameIds.end());
  std::sort(frameIdsOther.begin(), frameIdsOther.end());
  for (size_t i = 0; i < frameIds.size(); i++)
    if (frameIds[i] != frameIdsOther[i]) {
      return false;
    }
#undef FIELD_EQUAL
  return true;
}

bool PanoDefinition::validateInputMasks() const {
  for (readerid_t i = 0; i < numInputs(); ++i) {
    if (!getInput(i).validateMask()) {
      return false;
    }
  }
  return true;
}

bool PanoDefinition::validate(std::ostream& os) const {
  if (getWidth() <= 0) {
    os << "width must be strictly positive." << std::endl;
    return false;
  }
  if (getHeight() <= 0) {
    os << "height must be strictly positive." << std::endl;
    return false;
  }
  if (getHFOV() <= 0.0) {
    os << "The field of view must be strictly positive." << std::endl;
    return false;
  }
  if (getHFOV() < MIN_FOV) {
    os << "The horizontal field of view must be larger than " << MIN_FOV << " degrees. Got " << getHFOV() << "."
       << std::endl;
    return false;
  }
  switch (getProjection()) {
    case PanoProjection::Equirectangular:
      if (getHFOV() > 360.0) {
        os << "The horizontal field of view must be smaller than 360 degrees for equirect. Got " << getHFOV() << "."
           << std::endl;
        return false;
      }
      if (getVFOV() > 180.0) {
        os << "The vertical field of view must be smaller than 180 degrees for equirect. Got " << getVFOV()
           << ". Set height to a correct value and use padding if you want to fill a larger image size." << std::endl;
        return false;
      }
      break;
    case PanoProjection::FullFrameFisheye:
    case PanoProjection::CircularFisheye:
      if (getHFOV() > 360.0) {
        os << "The horizontal field of view must be smaller than 360 degrees for fisheye. Got " << getHFOV() << "."
           << std::endl;
        return false;
      }
      if (getVFOV() > 360.0) {
        os << "The vertical field of view must be smaller than 360 degrees for fisheye. Got " << getVFOV()
           << ". Set height to a correct value and use padding if you want to fill a larger image size." << std::endl;
        return false;
      }
      break;
    case PanoProjection::Stereographic:
      if (getHFOV() > 359.9) {
        os << "The horizontal field of view must be smaller than 359.9 degrees for stereographic. Got " << getHFOV()
           << "." << std::endl;
        return false;
      }
      if (getVFOV() > 359.9) {
        os << "The vertical field of view must be smaller than 359.9 degrees for stereographic. Got " << getVFOV()
           << ". Set height to a correct value and use padding if you want to fill a larger image size." << std::endl;
        return false;
      }
      break;
    case PanoProjection::Rectilinear:
      if (getHFOV() > 179.9) {
        os << "The horizontal field of view must be smaller than 180 degrees for rectilinear. Got " << getHFOV() << "."
           << std::endl;
        return false;
      }
      if (getVFOV() > 179.9) {
        os << "The vertical field of view must be smaller than 180 degrees for rectilinear. Got " << getVFOV()
           << ". Set height to a correct value and use padding if you want to fill a larger image size." << std::endl;
        return false;
      }
      break;
    case PanoProjection::EquiangularCubemap:
    case PanoProjection::Cubemap:
    case PanoProjection::Cylindrical:
      break;
  }
  if (numVideoInputs() > MAX_VIDEO_INPUTS) {
    os << "Only up to " << MAX_VIDEO_INPUTS << " video inputs are supported, got " << numVideoInputs() << std::endl;
    return false;
  }
  if (numOverlays() > MAX_OVERLAYS) {
    os << "Only up to " << MAX_OVERLAYS << " overlays are supported, got " << numOverlays() << std::endl;
    return false;
  }

  if (getSphereScale() <= 0.0) {
    os << "The sphere scale must be strictly positive." << std::endl;
    return false;
  }
  if (!getControlPointListDef().validate(os, numVideoInputs())) {
    os << "Invalid calibration control points." << std::endl;
    return false;
  }
  // rig presets are optional, check if they are present before validating
  if (hasCalibrationRigPresets() && !getCalibrationRigPresets().validate(os, numVideoInputs())) {
    os << "Invalid calibration rig presets." << std::endl;
    return false;
  }
  return true;
}

PanoDefinition::~PanoDefinition() { delete pimpl; }

const MergerMaskDefinition& PanoDefinition::getMergerMask() const {
  assert(pimpl->mergerMask);
  return *pimpl->mergerMask;
}

PanoDefinition::Pimpl::~Pimpl() {
  for (std::vector<InputDefinition*>::iterator it = inputs.begin(); it != inputs.end(); ++it) {
    delete *it;
  }
  for (std::vector<OverlayInputDefinition*>::iterator it = overlays.begin(); it != overlays.end(); ++it) {
    delete *it;
  }
  delete postprocessors;
  delete mergerMask;
  delete controlPointList;
  delete rigDefinition;
}

// Helper function to avoid code duplication
static readerid_t findInputIndexInArray(const Core::InputDefinition& idef,
                                        const std::vector<std::reference_wrapper<const InputDefinition>>& array,
                                        const std::string& failureMessage) {
  for (readerid_t i = 0; i < (readerid_t)array.size(); i++) {
    // VSA-7140: compare InputDefinition addresses instead of contents, some inputs can have identical contents
    if (&idef == &array[i].get()) {
      return i;
    }
  }
  // should not get here
  Logger::get(Logger::Error) << failureMessage << std::endl;
  std::abort();
  // keep the compiler happy
  return 0;
}

videoreaderid_t PanoDefinition::convertInputIndexToVideoInputIndex(readerid_t i) const {
  // i is expressed as an input number, find the same input as a video input number
  return findInputIndexInArray(getInput(i), getVideoInputs(), "Video input not found");
}

audioreaderid_t PanoDefinition::convertInputIndexToAudioInputIndex(readerid_t i) const {
  // i is expressed as an input number, find the same input as an audio input number
  return findInputIndexInArray(getInput(i), getAudioInputs(), "Audio input not found");
}

readerid_t PanoDefinition::convertVideoInputIndexToInputIndex(videoreaderid_t i) const {
  return findInputIndexInArray(getVideoInput(i), getInputs(), "Video input not found");
}

readerid_t PanoDefinition::convertAudioInputIndexToInputIndex(audioreaderid_t i) const {
  return findInputIndexInArray(getAudioInput(i), getInputs(), "Audio input not found");
}

const InputDefinition& PanoDefinition::getInput(readerid_t i) const {
  assert(0 <= i && i < (readerid_t)pimpl->inputs.size());
  return *pimpl->inputs[i];
}

InputDefinition& PanoDefinition::getInput(readerid_t i) {
  assert(0 <= i && i < (readerid_t)pimpl->inputs.size());
  return *pimpl->inputs[i];
}

// Helper template function to avoid code duplication
template <class ReturnType, class PanoType>
static ReturnType& getInputWithPredicate(PanoType& pano, readerid_t i,
                                         bool (InputDefinition::*predicateFunction)() const,
                                         const std::string& failureMessage) {
  // Go through inputs and decrease i on video ones, until we get i == 0
  for (readerid_t k = 0; k < pano.numInputs(); k++) {
    ReturnType& def = pano.getInput(k);
    if ((def.*predicateFunction)()) {
      if (i == 0) {
        return def;
      } else {
        --i;
      }
    }
  }
  // should not get here
  Logger::get(Logger::Error) << failureMessage << std::endl;
  std::abort();
}

const InputDefinition& PanoDefinition::getVideoInput(videoreaderid_t i) const {
  return getInputWithPredicate<const InputDefinition>(*this, i, &InputDefinition::getIsVideoEnabled,
                                                      "Video input not found");
}

InputDefinition& PanoDefinition::getVideoInput(videoreaderid_t i) {
  return getInputWithPredicate<InputDefinition>(*this, i, &InputDefinition::getIsVideoEnabled, "Video input not found");
}

const InputDefinition& PanoDefinition::getAudioInput(audioreaderid_t i) const {
  return getInputWithPredicate<const InputDefinition>(*this, i, &InputDefinition::getIsAudioEnabled,
                                                      "Audio input not found");
}

InputDefinition& PanoDefinition::getAudioInput(audioreaderid_t i) {
  return getInputWithPredicate<InputDefinition>(*this, i, &InputDefinition::getIsAudioEnabled, "Audio input not found");
}

// Helper template function to avoid code duplication
template <class ReturnType, class PanoType>
static std::vector<std::reference_wrapper<ReturnType>> getInputsWithPredicate(
    PanoType& pano, bool (InputDefinition::*predicateFunction)() const) {
  std::vector<std::reference_wrapper<ReturnType>> inputs;
  // Do not go through the pimpl inputs vector, go through virtual functions instead
  for (readerid_t i = 0; i < pano.numInputs(); i++) {
    ReturnType& idef = pano.getInput(i);
    if (predicateFunction == nullptr || (idef.*predicateFunction)()) {
      inputs.push_back(idef);
    }
  }
  return inputs;
}

std::vector<std::reference_wrapper<const InputDefinition>> PanoDefinition::getInputs() const {
  return getInputsWithPredicate<const InputDefinition>(*this, nullptr);
}

std::vector<std::reference_wrapper<InputDefinition>> PanoDefinition::getInputs() {
  return getInputsWithPredicate<InputDefinition>(*this, nullptr);
}

std::vector<std::reference_wrapper<const InputDefinition>> PanoDefinition::getVideoInputs() const {
  return getInputsWithPredicate<const InputDefinition>(*this, &InputDefinition::getIsVideoEnabled);
}

std::vector<std::reference_wrapper<InputDefinition>> PanoDefinition::getVideoInputs() {
  return getInputsWithPredicate<InputDefinition>(*this, &InputDefinition::getIsVideoEnabled);
}

std::vector<std::reference_wrapper<const InputDefinition>> PanoDefinition::getAudioInputs() const {
  return getInputsWithPredicate<const InputDefinition>(*this, &InputDefinition::getIsAudioEnabled);
}

std::vector<std::reference_wrapper<InputDefinition>> PanoDefinition::getAudioInputs() {
  return getInputsWithPredicate<InputDefinition>(*this, &InputDefinition::getIsAudioEnabled);
}

MergerMaskDefinition& PanoDefinition::getMergerMask() {
  assert(pimpl->mergerMask);
  return *pimpl->mergerMask;
}

bool PanoDefinition::getBlendingMaskEnabled() const { return getMergerMask().getEnabled(); }

void PanoDefinition::setBlendingMaskEnabled(const bool enabled) { getMergerMask().setEnabled(enabled); }

bool PanoDefinition::getBlendingMaskInterpolationEnabled() const { return getMergerMask().getInterpolationEnabled(); }

void PanoDefinition::setBlendingMaskInterpolationEnabled(const bool enabled) {
  getMergerMask().setInterpolationEnabled(enabled);
}

int64_t PanoDefinition::getBlendingMaskWidth() const { return getMergerMask().getWidth(); }

int64_t PanoDefinition::getBlendingMaskHeight() const { return getMergerMask().getHeight(); }

std::vector<frameid_t> PanoDefinition::getBlendingMaskFrameIds() const { return getMergerMask().getFrameIds(); }

void PanoDefinition::removeBlendingMaskFrameIds(const std::vector<frameid_t>& frameIds) {
  getMergerMask().removeFrameIds(frameIds);
}

std::pair<frameid_t, frameid_t> PanoDefinition::getBlendingMaskBoundedFrameIds(const frameid_t frameId) const {
  return getMergerMask().getInputIndexPixelData().getBoundedFrameIds(frameId);
}

const ControlPointListDefinition& PanoDefinition::getControlPointListDef() const {
  assert(pimpl->controlPointList);
  return *pimpl->controlPointList;
}

ControlPointListDefinition& PanoDefinition::getControlPointListDef() {
  assert(pimpl->controlPointList);
  return *pimpl->controlPointList;
}

std::vector<size_t> PanoDefinition::getMasksOrder() const {
  std::vector<size_t> masksOrder = getMergerMask().getMasksOrder();
  if (!(getMergerMask().getWidth() == getWidth() && getMergerMask().getHeight() == getHeight() &&
        masksOrder.size() == (size_t)numInputs())) {
    masksOrder.clear();
    for (readerid_t i = 0; i < numInputs(); i++) {
      masksOrder.push_back(i);
    }
  }
  return masksOrder;
}

int PanoDefinition::getBlendingMaskInputScaleFactor() const { return pimpl->mergerMask->getInputScaleFactor(); }

std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> PanoDefinition::getInputIndexPixelDataIfValid(
    const frameid_t frameId) const {
  if (!(getMergerMask().getWidth() == getWidth() && getMergerMask().getHeight() == getHeight() &&
        getMergerMask().getMasksOrder().size() == (size_t)numInputs())) {
    return std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>>();
  }
  return getMergerMask().getInputIndexPixelDataIfValid(frameId);
}

void PanoDefinition::insertInput(InputDefinition* inputDef, readerid_t i) { safeInsert(pimpl->inputs, inputDef, i); }

InputDefinition* PanoDefinition::popInput(readerid_t i) {
  bool isVideoInput = getInput(i).getIsVideoEnabled();
  videoreaderid_t removedVideoInputIndex = (isVideoInput) ? convertInputIndexToVideoInputIndex(i) : 0;
  InputDefinition* ret = safeRemove(pimpl->inputs, i);

  if (ret != nullptr && isVideoInput) {
    // if calibration control points is not empty,
    // remove any cntrol point referred to this input
    if (!this->getCalibrationControlPointList().empty()) {
      VideoStitch::Core::ControlPointList controlPoints = getCalibrationControlPointList();
      auto it = controlPoints.begin();
      while (it != controlPoints.end()) {
        // remove control points involving removed input
        if (it->index0 == removedVideoInputIndex || it->index1 == removedVideoInputIndex) {
          it = controlPoints.erase(it);
          continue;
        }

        // update control point input's index, according
        // to new inputs indexes
        if (it->index0 > removedVideoInputIndex) {
          --(it->index0);
        }

        if (it->index1 > removedVideoInputIndex) {
          --(it->index1);
        }

        ++it;
      }
      this->setCalibrationControlPointList(controlPoints);
    }
  }

  return ret;
}

bool PanoDefinition::removeInput(readerid_t i) {
  InputDefinition* ret = popInput(i);

  if (ret != nullptr) {
    delete ret;
    return true;
  }

  return false;
}

readerid_t PanoDefinition::numInputs() const { return (readerid_t)pimpl->inputs.size(); }

// Helper function to avoid code duplication
static readerid_t countInputs(const PanoDefinition& pano, bool (InputDefinition::*predicateFunction)() const) {
  readerid_t count = 0;
  for (readerid_t i = 0; i < pano.numInputs(); i++) {
    if ((pano.getInput(i).*predicateFunction)()) {
      ++count;
    }
  }
  return count;
}

videoreaderid_t PanoDefinition::numVideoInputs() const {
  return countInputs(*this, &InputDefinition::getIsVideoEnabled);
}

audioreaderid_t PanoDefinition::numAudioInputs() const {
  return countInputs(*this, &InputDefinition::getIsAudioEnabled);
}

double PanoDefinition::getHFovFromInputSources() const {
  std::set<double> setHFOV;
  // Do not go through the pimpl inputs vector, go through virtual functions instead
  for (readerid_t i = 0; i < numInputs(); i++) {
    const InputDefinition& idef = getInput(i);
    if (idef.getIsVideoEnabled()) {
      double hfov = idef.getGeometries().at(0).getEstimatedHorizontalFov(idef);
      setHFOV.insert(hfov);
    }
  }
  if (setHFOV.size() == 1) {
    return *(setHFOV.begin());
  } else if (setHFOV.empty()) {
    return PTV_DEFAULT_HFOV;
  } else {
    std::string errmsg =
        "Warning: Multiple hfov values defined. All the input sources must have the same horizontal field of view";
    VideoStitch::Logger::get(VideoStitch::Logger::Warning) << errmsg << std::endl;
    // Return the median
    auto iter = setHFOV.begin();
    std::advance(iter, setHFOV.size() / 2);
    return *(iter);
  }
}

template <class T>
static T checkAndGetSingleValueFromInputSources(T (InputDefinition::*thisgetter)() const, const PanoDefinition& pano,
                                                const T defaultValue, const std::string& errorMessage) {
  std::set<T> setOfValues;
  for (const InputDefinition& videoInput : pano.getVideoInputs()) {
    setOfValues.insert((videoInput.*thisgetter)());
  }
  if (setOfValues.empty()) {
    return defaultValue;
  } else if (setOfValues.size() == 1) {
    return *(setOfValues.begin());
  } else {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning) << errorMessage << std::endl;
    return *(setOfValues.begin());
  }
}

InputDefinition::Format PanoDefinition::getLensFormatFromInputSources() const {
  return checkAndGetSingleValueFromInputSources(
      &InputDefinition::getFormat, *this, InputDefinition::Format::FullFrameFisheye,
      "Warning: Inconsistent values for lens types. All inputs should have the same Projection parameter");
}

InputDefinition::LensModelCategory PanoDefinition::getLensModelCategoryFromInputSources() const {
  return checkAndGetSingleValueFromInputSources(
      &InputDefinition::getLensModelCategory, *this, InputDefinition::LensModelCategory::Legacy,
      "Warning: Inconsistent values for lens categories. All inputs should have the same lens category");
}

std::vector<std::reference_wrapper<const OverlayInputDefinition>> PanoDefinition::getOverlays() const {
  std::vector<std::reference_wrapper<const OverlayInputDefinition>> overlays;
  for (readerid_t i = 0; i < numOverlays(); i++) {
    const OverlayInputDefinition& idef = getOverlay(i);
    overlays.push_back(idef);
  }
  return overlays;
}

std::vector<std::reference_wrapper<OverlayInputDefinition>> PanoDefinition::getOverlays() {
  std::vector<std::reference_wrapper<OverlayInputDefinition>> overlays;
  for (readerid_t i = 0; i < numOverlays(); i++) {
    OverlayInputDefinition& idef = getOverlay(i);
    overlays.push_back(idef);
  }
  return overlays;
}

const OverlayInputDefinition& PanoDefinition::getOverlay(overlayreaderid_t i) const {
  assert(0 <= i && i < (overlayreaderid_t)pimpl->overlays.size());
  return *pimpl->overlays[i];
}

OverlayInputDefinition& PanoDefinition::getOverlay(overlayreaderid_t i) {
  assert(0 <= i && i < (overlayreaderid_t)pimpl->overlays.size());
  return *pimpl->overlays[i];
}

void PanoDefinition::insertOverlay(OverlayInputDefinition* overlayDef, overlayreaderid_t i) {
  safeInsert(pimpl->overlays, overlayDef, i);
}

OverlayInputDefinition* PanoDefinition::popOverlay(overlayreaderid_t i) {
  OverlayInputDefinition* ret = safeRemove(pimpl->overlays, i);
  return ret;
}

bool PanoDefinition::removeOverlay(overlayreaderid_t i) {
  OverlayInputDefinition* ret = popOverlay(i);

  if (ret != nullptr) {
    delete ret;
    return true;
  }

  return false;
}

overlayreaderid_t PanoDefinition::numOverlays() const { return (overlayreaderid_t)pimpl->overlays.size(); }

int64_t PanoDefinition::getWidth() const { return pimpl->width; }

int64_t PanoDefinition::getHeight() const { return pimpl->height; }

int64_t PanoDefinition::getLength() const { return pimpl->length; }

const char* PanoDefinition::getFormatName(const PanoProjection& fmt) { return getPanoProjectionName(fmt); }

PanoProjection PanoDefinition::getFormatFromName(const std::string& fmt) {
  PanoProjection proj(PanoProjection::Equirectangular);
  if (!getPanoProjectionFromName(fmt, proj)) {
    Logger::get(Logger::Warning) << "'" << fmt
                                 << "' output projection is not supported yet. Falling back to equirectangular."
                                 << std::endl;
  }
  return proj;
}

bool PanoDefinition::fromPTFormat(const char* ptFmt, PanoProjection* fmt) {
  std::string s(ptFmt);
  if (!s.compare("0")) {
    *fmt = PanoProjection(PanoProjection::Rectilinear);
    return true;
  } else if (!s.compare("1")) {
    *fmt = PanoProjection(PanoProjection::Cylindrical);
    return true;
  } else if (!s.compare("2")) {
    *fmt = PanoProjection(PanoProjection::Equirectangular);
    return true;
  } else if (!s.compare("3")) {
    *fmt = PanoProjection(PanoProjection::FullFrameFisheye);
    return true;
  } else if (!s.compare("4")) {
    *fmt = PanoProjection(PanoProjection::Stereographic);
    return true;
  }
  return false;
}

bool PanoDefinition::fromPTSFormat(const std::string& ptsFmt, PanoProjection* fmt) {
  if (!ptsFmt.compare("frectilinear")) {
    *fmt = PanoProjection(PanoProjection::Rectilinear);
    return true;
  } else if (!ptsFmt.compare("fcylindrical")) {
    *fmt = PanoProjection(PanoProjection::Cylindrical);
    return true;
  } else if (!ptsFmt.compare("fequirectangular")) {
    *fmt = PanoProjection(PanoProjection::Equirectangular);
    return true;
  } else if (!ptsFmt.compare("ffullframe")) {
    *fmt = PanoProjection(PanoProjection::FullFrameFisheye);
    return true;
  } else if (!ptsFmt.compare("fcircular")) {
    *fmt = PanoProjection(PanoProjection::CircularFisheye);
    return true;
  } else if (!ptsFmt.compare("fstereographic")) {
    *fmt = PanoProjection(PanoProjection::Stereographic);
    return true;
  } else if (!ptsFmt.compare("fstereographic_down")) {
    *fmt = PanoProjection(PanoProjection::Stereographic);
    return true;
  }
  return false;
}

double PanoDefinition::getHFOV() const { return pimpl->hFOV; }

double PanoDefinition::getVFOV() const {
  switch (getProjection()) {
    case PanoProjection::Rectilinear:
      // hfov = 2 * atan(w / 2f)  =>  2f = w / tan(hfov / 2)
      // vfov = 2 * atan(h / 2f)
      return 2.0 * atan(((double)getHeight() * tan(getHFOV() / 2.0)) / (double)getWidth());
    case PanoProjection::Cylindrical:
    case PanoProjection::Equirectangular:
    case PanoProjection::FullFrameFisheye:
    case PanoProjection::Stereographic:
    case PanoProjection::CircularFisheye:
      return ((double)getHeight() * getHFOV()) / (double)getWidth();
    case PanoProjection::EquiangularCubemap:
    case PanoProjection::Cubemap:
      return M_PI;
  }
  return 0;
}

double PanoDefinition::getCalibrationCost() const { return pimpl->calibrationCost; }

double PanoDefinition::getCalibrationInitialHFOV() const { return pimpl->calibrationInitialHFOV; }

void PanoDefinition::setCalibrationCost(double cost) { pimpl->calibrationCost = cost; }

void PanoDefinition::setCalibrationInitialHFOV(double hfov) { pimpl->calibrationInitialHFOV = hfov; }

void PanoDefinition::setCalibrationControlPointList(const ControlPointList& list) {
  getControlPointListDef().setCalibrationControlPointList(list);
}

const ControlPointList& PanoDefinition::getCalibrationControlPointList() const {
  return getControlPointListDef().getCalibrationControlPointList();
}

void PanoDefinition::setCalibrationRigPresets(RigDefinition* rigDef) {
  delete pimpl->rigDefinition;
  pimpl->rigDefinition = rigDef;
}

const RigDefinition& PanoDefinition::getCalibrationRigPresets() const {
  assert(pimpl->rigDefinition);
  return *pimpl->rigDefinition;
}

std::string PanoDefinition::getCalibrationRigPresetsName() const { return pimpl->rigDefinition->getName(); }

bool PanoDefinition::hasCalibrationRigPresets() const {
  return (pimpl->rigDefinition->getRigCameraDefinitionCount() != 0);
}

bool PanoDefinition::isRigPresetCompatible(const VideoStitch::Ptv::Value* rigValue) const {
  if (!rigValue->has("rig") || !rigValue->has("cameras") || !rigValue->has("rig")->has("rigcameras")) {
    return false;
  }

  // First check the number of cameras
  const auto& nbCameras = numVideoInputs();
  const auto& rigCameras = rigValue->has("rig")->has("rigcameras")->asList();
  if (rigCameras.size() != (size_t)nbCameras || nbCameras == 0) {
    return false;
  }

  // Then the cameras could be in the wrong order
  // so we check that we have the same number of cameras for each resolution
  std::map<std::pair<int64_t, int64_t>, int> presetCameraResolutions;
  std::map<std::pair<int64_t, int64_t>, int> panoCameraResolutions;
  const auto& presetCameras = rigValue->has("cameras")->asList();
  const auto& panoCameras = getVideoInputs();

  for (auto index = 0; index < nbCameras; ++index) {
    ++panoCameraResolutions[std::make_pair(panoCameras.at(index).get().getWidth(),
                                           panoCameras.at(index).get().getHeight())];
    const std::string& cameraName = rigCameras.at(index)->has("camera")->asString();
    for (auto presetCamera : presetCameras) {
      if (cameraName == presetCamera->has("name")->asString() && presetCamera->has("width") &&
          presetCamera->has("height")) {
        ++presetCameraResolutions[std::make_pair(presetCamera->has("width")->asInt(),
                                                 presetCamera->has("height")->asInt())];
        break;
      }
    }
  }
  return presetCameraResolutions == panoCameraResolutions;
}

bool PanoDefinition::hasBeenSynchronized() const {
  for (const InputDefinition& inputDef : getVideoInputs()) {
    if (inputDef.getFrameOffset() != 0) {
      return true;
    }
  }
  return false;
}

void PanoDefinition::setHasBeenCalibrationDeshuffled(const bool deshuffled) {
  pimpl->calibrationDeshuffled = deshuffled;
}

bool PanoDefinition::hasBeenCalibrationDeshufled() const { return pimpl->calibrationDeshuffled; }

bool PanoDefinition::hasCalibrationControlPoints() const { return !getCalibrationControlPointList().empty(); }

bool PanoDefinition::hasBeenCalibrated() const {
  if (numVideoInputs() == 1) {
    return false;
  }
  bool repeated = false;
  for (int i = 0; i < numVideoInputs(); ++i) {
    for (int j = i + 1; j < numVideoInputs(); ++j) {
      if (getInput(i).getGeometries().at(0).hasSameExtrinsics(getInput(j).getGeometries().at(0))) {
        repeated = true;
        break;
      }
    }
    if (repeated) {
      break;
    }
  }
  return !repeated;
}

bool PanoDefinition::photometryHasBeenCalibrated() const {
  for (const InputDefinition& videoInput : getVideoInputs()) {
    if (videoInput.getEmorA() != 0.0 || videoInput.getEmorB() != 0.0 || videoInput.getEmorC() != 0.0 ||
        videoInput.getEmorD() != 0.0 || videoInput.getEmorE() != 0.0 || videoInput.getVignettingCoeff1() != 0.0 ||
        videoInput.getVignettingCoeff2() != 0.0 || videoInput.getVignettingCoeff3() != 0.0) {
      return true;
    }
  }
  return false;
}

bool PanoDefinition::hasTranslations() const {
  for (const auto& idef : getVideoInputs()) {
    if (idef.get().getGeometries().at(0).hasTranslation()) {
      return true;
    }
  }

  return false;
}

double PanoDefinition::computeMinimumRigSphereRadius() const {
  if (!hasTranslations()) {
    return 0.;
  }

  float minRadius = std::numeric_limits<float>::max();
  for (const auto& idef : getVideoInputs()) {
    const auto& geometry = idef.get().getGeometries().at(0);
    if (geometry.hasTranslation()) {
      const std::unique_ptr<TransformStack::GeoTransform> geoTransform(
          TransformStack::GeoTransform::create(*this, idef));
      minRadius = std::min(geoTransform->computeInputMinimumRigSphereRadius(idef, 0), minRadius);
    }
  }
  return minRadius;
}

void PanoDefinition::setWidth(uint64_t w) { pimpl->width = w; }

void PanoDefinition::setHeight(uint64_t h) { pimpl->height = h; }

void PanoDefinition::setLength(uint64_t l) { pimpl->length = l; }

void PanoDefinition::setProjection(PanoProjection format) { pimpl->projection = format; }

void PanoDefinition::setHFOV(double hFov) { pimpl->hFOV = hFov; }

void PanoDefinition::setVFOV(double vFov) {
  switch (getProjection()) {
    case PanoProjection::Rectilinear:
      pimpl->hFOV = 2.0 * atan(((double)getWidth() * tan(vFov / 2.0)) / (double)getHeight());
      break;
    case PanoProjection::Cylindrical:
    case PanoProjection::Equirectangular:
    case PanoProjection::FullFrameFisheye:
    case PanoProjection::Stereographic:
    case PanoProjection::CircularFisheye:
      pimpl->hFOV = ((double)getWidth() * vFov) / (double)getHeight();
      break;
    case PanoProjection::Cubemap:
    case PanoProjection::EquiangularCubemap:
      break;
  }
}

const Ptv::Value* PanoDefinition::getPostprocessors() const { return pimpl->postprocessors; }

bool PanoDefinition::getPrecomputedCoordinateBuffer() const { return pimpl->precomputedCoodinateBuffer; }

void PanoDefinition::setPrecomputedCoordinateBuffer(const bool b) { pimpl->precomputedCoodinateBuffer = b; }

double PanoDefinition::getPrecomputedCoordinateShrinkFactor() const { return pimpl->precomputedCoodinateShrinkFactor; }

void PanoDefinition::setPrecomputedCoordinateShrinkFactor(const double b) {
  pimpl->precomputedCoodinateShrinkFactor = b;
}

namespace {
/**
 * Returns true if a format is allowed to wrap.
 */
bool canWrap(PanoProjection fmt) {
  switch (fmt) {
    case PanoProjection::Rectilinear:
      return false;
    case PanoProjection::Cylindrical:
      return true;
    case PanoProjection::Equirectangular:
      return true;
    case PanoProjection::FullFrameFisheye:
      return false;
    case PanoProjection::Stereographic:
      return false;
    case PanoProjection::CircularFisheye:
      return false;
    case PanoProjection::Cubemap:
    case PanoProjection::EquiangularCubemap:
      return false;
  }
  return false;
}
}  // namespace

void PanoDefinition::computeOptimalPanoSize(unsigned& width, unsigned& height) const {
  double minDist = std::numeric_limits<double>::max();
  for (readerid_t i = 0; i < numInputs(); ++i) {
    const std::unique_ptr<InputDefinition> fakeInput(getInput(i).clone());
    const std::unique_ptr<TransformStack::GeoTransform> transform(
        TransformStack::GeoTransform::create(*this, *fakeInput));

    const CenterCoords2 cIn = transform->mapInputToPanorama(*fakeInput, CenterCoords2(0.0f, 0.0f), 0);
    const CenterCoords2 uIn = transform->mapInputToPanorama(*fakeInput, CenterCoords2(1.0f, 1.0f), 0);

    // make sure projected points are in the pano
    if (cIn.x < getWidth() / 2 && cIn.y < getHeight() / 2 && cIn.x > -getWidth() / 2 && cIn.y > -getHeight() / 2 &&
        uIn.x < getWidth() / 2 && uIn.y < getHeight() / 2 && uIn.x > -getWidth() / 2 && uIn.y > -getHeight() / 2) {
      double dist = sqrt((uIn.x - cIn.x) * (uIn.x - cIn.x) + (uIn.y - cIn.y) * (uIn.y - cIn.y));
      if (dist < minDist) {
        minDist = dist;
      }
    }
  }

  if (minDist == std::numeric_limits<double>::max()) {
    // none of input image centers lie inside pano bounds when projected to the output
    // something wrong with the calibration/configuration, will not suggest optimal size
    width = (unsigned)getWidth();
    height = (unsigned)getHeight();
    return;
  }

  double scaleFactor = M_SQRT2 / minDist;
  double ratio = (double)getWidth() / (double)getHeight();

  // most kernels use workgroups of size 16
  // lets round up a bit to get full GPU utilisation
  auto roundUpToMultipleOf16 = [](double val) -> unsigned { return (((unsigned)val + 16 - 1) / 16) * 16; };

  height = roundUpToMultipleOf16((double)getHeight() * scaleFactor);
  width = (unsigned)(height * ratio);
}

GENCURVEFUNCTIONS(PanoDefinition, Curve, RedCB, redCB, PTV_DEFAULT_PANODEF_REDCB)
GENCURVEFUNCTIONS(PanoDefinition, Curve, GreenCB, greenCB, PTV_DEFAULT_PANODEF_GREENCB)
GENCURVEFUNCTIONS(PanoDefinition, Curve, BlueCB, blueCB, PTV_DEFAULT_PANODEF_BLUECB)
GENCURVEFUNCTIONS(PanoDefinition, Curve, ExposureValue, exposureValue, PTV_DEFAULT_PANODEF_EXPOSURE)
GENCURVEFUNCTIONS(PanoDefinition, QuaternionCurve, Stabilization, stabilizationCurve, Quaternion<double>())
GENCURVEFUNCTIONS(PanoDefinition, QuaternionCurve, GlobalOrientation, orientationCurve, Quaternion<double>())
GENCURVEFUNCTIONS(PanoDefinition, Curve, StabilizationYaw, stabilizationYawCurve, 0.0)
GENCURVEFUNCTIONS(PanoDefinition, Curve, StabilizationPitch, stabilizationPitchCurve, 0.0)
GENCURVEFUNCTIONS(PanoDefinition, Curve, StabilizationRoll, stabilizationRollCurve, 0.0)

namespace {
Core::QuaternionCurve* ypr2quaternion(Core::Curve* yaw, Core::Curve* pitch, Core::Curve* roll) {
  // let's just suppose the yaw curve contains all the keyframes.
  // it's wrong, but we're in best effort mode here.
  Core::Spline* ys = yaw->splines();
  Core::SphericalSpline* firstSpline = NULL;
  if (ys != NULL) {
    firstSpline = Core::SphericalSpline::point(
        ys->end.t, Quaternion<double>::fromEulerZXY(degToRad(ys->end.v), degToRad(pitch->at(ys->end.t)),
                                                    degToRad(roll->at(ys->end.t))));
    Core::SphericalSpline* spline = firstSpline;
    while (ys->next != NULL) {
      ys = ys->next;
      ys->getType() == Core::Spline::LineType
          ? spline = spline->lineTo(
                ys->end.t, Quaternion<double>::fromEulerZXY(degToRad(ys->end.v), degToRad(pitch->at(ys->end.t)),
                                                            degToRad(roll->at(ys->end.t)))
                               .normalize())
          : spline = spline->cubicTo(
                ys->end.t, Quaternion<double>::fromEulerZXY(degToRad(ys->end.v), degToRad(pitch->at(ys->end.t)),
                                                            degToRad(roll->at(ys->end.t)))
                               .normalize());
    }
  }
  delete yaw;
  delete pitch;
  delete roll;
  return new Core::QuaternionCurve(firstSpline);
}

/**
 * Returns the inputs PTVs as a list, or NULL on error. Ownership is retained.
 */
const std::vector<Ptv::Value*>* getInputsPtv(const Ptv::Value& value) {
  const Ptv::Value* var = value.has("inputs");
  if (!Parse::checkVar("PanoDefinition", "inputs", var, true)) {
    return nullptr;
  }
  if (!Parse::checkType("inputs", *var, Ptv::Value::LIST)) {
    return nullptr;
  }
  return &var->asList();
}

bool parseInputs(const Ptv::Value& value, std::vector<Core::InputDefinition*>& inputDefs) {
  const std::vector<Ptv::Value*>* inputs = getInputsPtv(value);
  if (!inputs) {
    return false;
  }
  for (size_t i = 0; i < inputs->size(); ++i) {
    Core::InputDefinition* input = Core::InputDefinition::create(*(*inputs)[i]);
    if (!input) {
      deleteAll(inputDefs);
      return false;
    }
    inputDefs.push_back(input);
  }
  return true;
}

/**
 * Returns the overlays PTVs as a list, or NULL on error. Ownership is retained.
 */
const std::vector<Ptv::Value*>* getOverlaysPtv(const Ptv::Value& value) {
  const Ptv::Value* var = value.has("overlays");
  if (!Parse::checkVar("PanoDefinition", "overlays", var, false)) {
    return nullptr;
  }
  if (!Parse::checkType("overlays", *var, Ptv::Value::LIST)) {
    return nullptr;
  }
  return &var->asList();
}

bool parseOverlays(const Ptv::Value& value, std::vector<Core::OverlayInputDefinition*>& overlayDefs) {
  const std::vector<Ptv::Value*>* overlays = getOverlaysPtv(value);
  if (!overlays) {
    return false;
  }
  for (size_t i = 0; i < overlays->size(); ++i) {
    Core::OverlayInputDefinition* overlay = Core::OverlayInputDefinition::create(*(*overlays)[i]);
    if (!overlay) {
      deleteAll(overlayDefs);
      return false;
    }
    overlayDefs.push_back(overlay);
  }
  return true;
}

}  // namespace

Core::PanoDefinition* Core::PanoDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("PanoDefinition", value, Ptv::Value::OBJECT)) {
    return nullptr;
  }
  std::unique_ptr<PanoDefinition> res(new PanoDefinition());
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return NULL;                          \
  }
  PROPAGATE_NOK(Parse::populateInt("PanoDefinition", value, "width", res->pimpl->width, true));
  PROPAGATE_NOK(Parse::populateInt("PanoDefinition", value, "height", res->pimpl->height, true));
  PROPAGATE_NOK(Parse::populateDouble("PanoDefinition", value, "hfov", res->pimpl->hFOV, true));
  std::string proj;
  PROPAGATE_NOK(Parse::populateString("PanoDefinition", value, "proj", proj, false));
  getPanoProjectionFromName(proj, res->pimpl->projection);
#undef PROPAGATE_NOK

  // Populate inputs:
  {
    std::vector<Core::InputDefinition*> inputs;
    if (!parseInputs(value, inputs)) {
      return nullptr;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      res->pimpl->inputs.push_back(inputs[i]);
    }
  }
  // Populate overlays:
  { parseOverlays(value, res->pimpl->overlays); }
  // Populate mergerMask:
  {
    MergerMaskDefinition* mergerMask = MergerMaskDefinition::create(value);
    if (!mergerMask) {
      return nullptr;
    }
    res->pimpl->mergerMask = mergerMask;
  }
  // Populate controlPointList:
  {
    Potential<ControlPointListDefinition> controlPointList = ControlPointListDefinition::create(value);
    if (!controlPointList.ok()) {
      return nullptr;
    }
    res->pimpl->controlPointList = controlPointList.release();
  }
  // Populate optional rig presets
  {
    // Load cameras presets
    std::map<std::string, std::shared_ptr<Core::CameraDefinition>> cameras_map;
    const Ptv::Value* val_list_cameras = value.has("cameras");
    if (val_list_cameras && val_list_cameras->getType() == Ptv::Value::LIST) {
      std::vector<Ptv::Value*> list_cameras = val_list_cameras->asList();

      // Loop over list of camera presets
      for (auto f : list_cameras) {
        std::shared_ptr<Core::CameraDefinition> cam(Core::CameraDefinition::create(*f));
        if (cam.get()) {
          cameras_map[cam->getName()] = cam;
        } else {
          Logger::get(Logger::Error) << "Invalid camera presets definition in pano definition" << std::endl;
          return nullptr;
        }
      }
    }
    // If camera presets were loaded, load the rig presets
    if (!cameras_map.empty()) {
      // Load rig presets
      const Ptv::Value* val_rig = value.has("rig");
      if (val_rig && val_rig->getType() == Ptv::Value::OBJECT) {
        res->pimpl->rigDefinition = Core::RigDefinition::create(cameras_map, *val_rig);
      }
      if (res->pimpl->rigDefinition == nullptr) {
        Logger::get(Logger::Error) << "Invalid or missing rig presets definition in pano definition" << std::endl;
        return nullptr;
      }
    }
    if (res->pimpl->rigDefinition == nullptr) {
      res->pimpl->rigDefinition = new RigDefinition();
    }
  }

  if (!res->parseOrientationCurves(value).ok()) {
    return nullptr;
  }

  if (!res->parseExposureCurves(value).ok()) {
    return nullptr;
  }

  // Postprocessors:
  {
    const Ptv::Value* var = value.has("postprocessors");
    if (var) {
      res->pimpl->postprocessors = var->clone();
    }
  }

  // Optional values:
#define PROPAGATE_WRONGTYPE(call)                \
  if (call == Parse::PopulateResult_WrongType) { \
    return NULL;                                 \
  }
  PROPAGATE_WRONGTYPE(Parse::populateBool("PanoDefinition", value, "wrap", res->pimpl->wrap, false));
  PROPAGATE_WRONGTYPE(Parse::populateBool("PanoDefinition", value, "precomputed_coordinate_buffer",
                                          res->pimpl->precomputedCoodinateBuffer, false));
  PROPAGATE_WRONGTYPE(Parse::populateDouble("PanoDefinition", value, "precomputed_coordinate_shrink_factor",
                                            res->pimpl->precomputedCoodinateShrinkFactor, false));
  PROPAGATE_WRONGTYPE(Parse::populateInt("PanoDefinition", value, "crop_left", res->pimpl->cropLeft, false));
  PROPAGATE_WRONGTYPE(Parse::populateInt("PanoDefinition", value, "crop_right", res->pimpl->cropRight, false));
  PROPAGATE_WRONGTYPE(Parse::populateInt("PanoDefinition", value, "crop_top", res->pimpl->cropTop, false));
  PROPAGATE_WRONGTYPE(Parse::populateInt("PanoDefinition", value, "crop_bottom", res->pimpl->cropBottom, false));
  PROPAGATE_WRONGTYPE(
      Parse::populateDouble("PanoDefinition", value, "calibration_cost", res->pimpl->calibrationCost, false));
#undef PROPAGATE_WRONGTYPE

  if (Parse::populateDouble("PanoDefinition", value, "spherescale", res->pimpl->sphereScale, false) !=
      Parse::PopulateResult_Ok) {
    res->pimpl->sphereScale = PTV_DEFAULT_PANODEF_SPHERE_SCALE;
  }

  if (Parse::populateInt("PanoDefinition", value, "length", res->pimpl->length, false) != Parse::PopulateResult_Ok) {
    res->pimpl->length = PTV_DEFAULT_PANODEF_LENGTH;
  }

  if (Parse::populateDouble("PanoDefinition", value, "calibration_cost", res->pimpl->calibrationCost, false) !=
      Parse::PopulateResult_Ok) {
    res->pimpl->calibrationCost = -1.0;
  }

  std::stringstream errors;
  if (!res->validate(errors)) {
    Logger::get(Logger::Error) << errors.str();
    return nullptr;
  }
  return res.release();
}

Potential<Core::PanoDefinition> Core::PanoDefinition::createStereo(const Ptv::Value& panoDiff) const {
  // Make sure panoDiff is an object.
  if (!Parse::checkType("PanoDefinition", panoDiff, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Invalid panorama definition type"};
  }
  std::unique_ptr<Core::PanoDefinition> newPano(this->clone());

  // Input overloads.
  {
    const std::vector<Ptv::Value*>* inputs = getInputsPtv(panoDiff);
    if (inputs) {
      if (inputs->size() != (size_t)newPano->numInputs()) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot merge configurations. Mismatch in the number of inputs"};
      }
      for (readerid_t i = 0; i < newPano->numInputs(); ++i) {
        FAIL_RETURN(newPano->getInput(i).applyDiff(*(*inputs)[i], false));
      }
    }
  }

  // Only some pano parameters can be overloaded.
  FAIL_CAUSE(newPano->parseOrientationCurves(panoDiff), Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
             "Could not parse orientation curves");
  FAIL_CAUSE(newPano->parseExposureCurves(panoDiff), Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
             "Could not parse exposure curves");

  std::stringstream errors;
  if (!newPano->validate(errors)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Unable to create valid panorama configuration: " + errors.str()};
  }
  return Potential<Core::PanoDefinition>(newPano.release());
}

Ptv::Value* Core::PanoDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("width", new Parse::JsonValue((int)getWidth()));
  res->push("height", new Parse::JsonValue((int)getHeight()));
  if (getLength() != PTV_DEFAULT_PANODEF_LENGTH) {
    res->push("length", new Parse::JsonValue((int)getLength()));
  }

  res->push("hfov", new Parse::JsonValue(getHFOV()));

  res->push("proj", new Parse::JsonValue(getPanoProjectionName(getProjection())));

  if (getPrecomputedCoordinateBuffer()) {
    res->push("precomputed_coordinate_buffer", new Parse::JsonValue(getPrecomputedCoordinateBuffer()));
    res->push("precomputed_coordinate_shrink_factor", new Parse::JsonValue(getPrecomputedCoordinateShrinkFactor()));
  }

  res->push("global_orientation", getGlobalOrientation().serialize());
  res->push("stabilization", getStabilization().serialize());
  res->push("ev", getExposureValue().serialize());
  res->push("red_corr", getRedCB().serialize());
  res->push("green_corr", getGreenCB().serialize());
  res->push("blue_corr", getBlueCB().serialize());

  res->push("calibration_cost", new Parse::JsonValue(getCalibrationCost()));

  if (getSphereScale() != PTV_DEFAULT_PANODEF_SPHERE_SCALE) {
    res->push("spherescale", new Parse::JsonValue(getSphereScale()));
  }

  // Inputs:
  Ptv::Value* jsonInputs = new Parse::JsonValue((void*)NULL);
  for (readerid_t i = 0; i < numInputs(); ++i) {
    jsonInputs->asList().push_back(getInput(i).serialize());
  }
  res->push("inputs", jsonInputs);

  // Overlays:
  if (numOverlays() > 0) {
    Ptv::Value* jsonOverlays = new Parse::JsonValue((void*)NULL);
    for (readerid_t i = 0; i < numOverlays(); ++i) {
      jsonOverlays->asList().push_back(getOverlay(i).serialize());
    }
    res->push("overlays", jsonOverlays);
  }

  // MergerMask:
  res->push("merger_mask", getMergerMask().serialize());

  // ControlPointList:
  res->push("calibration_control_points", getControlPointListDef().serialize());

  // Optional rig presets:
  if (hasCalibrationRigPresets()) {
    // Cameras presets
    Ptv::Value* listCameras = Ptv::Value::emptyObject();
    for (auto it : getCalibrationRigPresets().getRigCameraDefinitionMap()) {
      listCameras->asList().push_back(it.second->serialize());
    }
    res->push("cameras", listCameras);

    // Rig presets
    res->push("rig", getCalibrationRigPresets().serialize());
  }

  // And postprocessors:
  if (getPostprocessors()) {
    res->push("postprocessors", getPostprocessors()->clone());
  }

  return res;
}

Status Core::PanoDefinition::parseOrientationCurves(const Ptv::Value& value) {
  // Legacy code to handle old orientation curves
  {
    Curve* yawCurve = NULL;
    Curve* pitchCurve = NULL;
    Curve* rollCurve = NULL;
    Curve* stabYawCurve = NULL;
    Curve* stabPitchCurve = NULL;
    Curve* stabRollCurve = NULL;
    {
      const Ptv::Value* var = value.has("global_yaw");
      if (var) {
        yawCurve = Core::Curve::create(*var);
        if (!yawCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse 'global_yaw' curve"};
        }
      }
    }
    {
      const Ptv::Value* var = value.has("global_pitch");
      if (var) {
        pitchCurve = Core::Curve::create(*var);
        if (!pitchCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse 'global_pitch' curve"};
        }
      }
    }
    {
      const Ptv::Value* var = value.has("global_roll");
      if (var) {
        rollCurve = Core::Curve::create(*var);
        if (!rollCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse 'global_roll' curve"};
        }
      }
    }
    {
      const Ptv::Value* var = value.has("stabilization_yaw");
      if (var) {
        stabYawCurve = Core::Curve::create(*var);
        if (!stabYawCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                  "Cannot parse 'stabilization_yaw' curve"};
        }
      }
    }
    {
      const Ptv::Value* var = value.has("stabilization_pitch");
      if (var) {
        stabPitchCurve = Core::Curve::create(*var);
        if (!stabPitchCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                  "Cannot parse 'stabilization_pitch' curve"};
        }
      }
    }
    {
      const Ptv::Value* var = value.has("stabilization_roll");
      if (var) {
        stabRollCurve = Core::Curve::create(*var);
        if (!stabRollCurve) {
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                  "Cannot parse 'stabilization_roll' curve"};
        }
      }
    }

    if (yawCurve && pitchCurve && rollCurve) {
      replaceGlobalOrientation(ypr2quaternion(yawCurve, pitchCurve, rollCurve));
    }
    if (stabYawCurve && stabPitchCurve && stabRollCurve) {
      replaceStabilization(ypr2quaternion(stabYawCurve, stabPitchCurve, stabRollCurve));
    }
  }

  // new orientation curves
  {
    const Ptv::Value* var = value.has("stabilization");
    if (var) {
      QuaternionCurve* curve = Core::QuaternionCurve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse 'stabilization' curve"};
      }
      replaceStabilization(curve);
      Curve *yaw = NULL, *pitch = NULL, *roll = NULL;
      Core::toEuler(*curve, &yaw, &pitch, &roll);
      replaceStabilizationYaw(yaw);
      replaceStabilizationPitch(pitch);
      replaceStabilizationRoll(roll);
    }
  }
  {
    const Ptv::Value* var = value.has("global_orientation");
    if (var) {
      QuaternionCurve* curve = Core::QuaternionCurve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse 'global_orientation' curve"};
      }
      replaceGlobalOrientation(curve);
    }
  }
  return Status::OK();
}

Status Core::PanoDefinition::parseExposureCurves(const Ptv::Value& value) {
  {
    const Ptv::Value* var = value.has("ev");
    if (var) {
      Curve* curve = Core::Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse exposure curve ('ev')"};
      }
      replaceExposureValue(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("red_corr");
    if (var) {
      Curve* curve = Core::Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse red correction curve ('red_corr')"};
      }
      replaceRedCB(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("green_corr");
    if (var) {
      Curve* curve = Core::Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse green correction curve ('green_corr')"};
      }
      replaceGreenCB(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("blue_corr");
    if (var) {
      Curve* curve = Core::Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse blue correction curve ('blue_corr')"};
      }
      replaceBlueCB(curve);
    }
  }
  return Status::OK();
}

PanoProjection PanoDefinition::getProjection() const { return pimpl->projection; }

double PanoDefinition::getSphereScale() const { return pimpl->sphereScale; }

void PanoDefinition::setSphereScale(double scale) { pimpl->sphereScale = scale; }

void PanoDefinition::resetExposure() {
  resetExposureValue();
  resetBlueCB();
  resetGreenCB();
  resetRedCB();
  for (readerid_t i = 0; i < numInputs(); i++) {
    getInput(i).resetExposure();
  }
}

void PanoDefinition::resetCalibration() {
  // Remove the rig and cameras
  if (pimpl->rigDefinition) {
    pimpl->rigDefinition->removeRigCameraDefinition();
  }
  // Remove the control points
  setCalibrationControlPointList(ControlPointList());
  for (InputDefinition& input : getVideoInputs()) {
    input.resetGeometries(PTV_DEFAULT_HFOV);
  }
}

}  // namespace Core
}  // namespace VideoStitch
