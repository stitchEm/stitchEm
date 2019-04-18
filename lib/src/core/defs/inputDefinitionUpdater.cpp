// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/inputDefinitionUpdater.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Core {
InputDefinitionUpdater::InputDefinitionUpdater(const InputDefinition& inputDefinition)
    : inputDefinition(inputDefinition.clone()) {}

InputDefinitionUpdater::InputDefinitionUpdater(InputDefinition* inputDefinition) : inputDefinition(inputDefinition) {}

InputDefinition* InputDefinitionUpdater::clone() const { return inputDefinition->clone(); }

Ptv::Value* InputDefinitionUpdater::serialize() const { return inputDefinition->serialize(); }

bool InputDefinitionUpdater::operator==(const InputDefinition& other) const {
  return inputDefinition->operator==(other);
}

bool InputDefinitionUpdater::validate(std::ostream& os) const { return inputDefinition->validate(os); }

std::string InputDefinitionUpdater::getDisplayName() const { return inputDefinition->getDisplayName(); }

const Ptv::Value& InputDefinitionUpdater::getReaderConfig() const { return inputDefinition->getReaderConfig(); }

const std::string& InputDefinitionUpdater::getMaskData() const { return inputDefinition->getMaskData(); }

bool InputDefinitionUpdater::deletesMaskedPixels() const { return inputDefinition->deletesMaskedPixels(); }

const InputDefinition::MaskPixelData& InputDefinitionUpdater::getMaskPixelData() const {
  return inputDefinition->getMaskPixelData();
}

const unsigned char* InputDefinitionUpdater::getMaskPixelDataIfValid() const {
  return inputDefinition->getMaskPixelDataIfValid();
}

bool InputDefinitionUpdater::validateMask() const { return inputDefinition->validateMask(); }

InputDefinition::group_t InputDefinitionUpdater::getGroup() const { return inputDefinition->getGroup(); }

void InputDefinitionUpdater::setGroup(InputDefinition::group_t group) {
  PRESERVE_ACTION(setGroup, inputDefinition, group);
}

int64_t InputDefinitionUpdater::getWidth() const { return inputDefinition->getWidth(); }

int64_t InputDefinitionUpdater::getHeight() const { return inputDefinition->getHeight(); }

void InputDefinitionUpdater::setWidth(int64_t int64) { PRESERVE_ACTION(setWidth, inputDefinition, int64); }

void InputDefinitionUpdater::setHeight(int64_t int64) { PRESERVE_ACTION(setHeight, inputDefinition, int64); }

int64_t InputDefinitionUpdater::getCroppedWidth() const { return inputDefinition->getCroppedWidth(); }

int64_t InputDefinitionUpdater::getCroppedHeight() const { return inputDefinition->getCroppedHeight(); }

bool InputDefinitionUpdater::getUseMeterDistortion() const { return inputDefinition->getUseMeterDistortion(); }

InputDefinition::Format InputDefinitionUpdater::getFormat() const { return inputDefinition->getFormat(); }

void InputDefinitionUpdater::setFormat(InputDefinition::Format format) {
  PRESERVE_ACTION(setFormat, inputDefinition, format);
}

InputDefinition::LensModelCategory InputDefinitionUpdater::getLensModelCategory() const {
  return inputDefinition->getLensModelCategory();
}

bool InputDefinitionUpdater::hasCroppedArea() const { return inputDefinition->hasCroppedArea(); }

frameid_t InputDefinitionUpdater::getFrameOffset() const { return inputDefinition->getFrameOffset(); }

double InputDefinitionUpdater::getSynchroCost() const { return inputDefinition->getSynchroCost(); }

int InputDefinitionUpdater::getStack() const { return inputDefinition->getStack(); }

void InputDefinitionUpdater::setStack(int value) { PRESERVE_ACTION(setStack, inputDefinition, value); }

void InputDefinitionUpdater::setFilename(const std::string& fileName) {
  PRESERVE_ACTION(setFilename, inputDefinition, fileName);
}

void InputDefinitionUpdater::setReaderConfig(Ptv::Value* config) {
  PRESERVE_ACTION_CLONEABLE(setReaderConfig, inputDefinition, , , Ptv::Value, config);
}

void InputDefinitionUpdater::setMaskData(const std::string& maskData) {
  PRESERVE_ACTION(setMaskData, inputDefinition, maskData);
}

void InputDefinitionUpdater::setDeletesMaskedPixels(bool value) {
  PRESERVE_ACTION(setDeletesMaskedPixels, inputDefinition, value);
}

bool InputDefinitionUpdater::setMaskPixelData(const char* buffer, uint64_t maskWidth, uint64_t maskHeight) {
  return inputDefinition->setMaskPixelData(buffer, maskWidth, maskHeight);
}

void InputDefinitionUpdater::setFrameOffset(int fo) { PRESERVE_ACTION(setFrameOffset, inputDefinition, fo); }

void InputDefinitionUpdater::setSynchroCost(double cost) { PRESERVE_ACTION(setSynchroCost, inputDefinition, cost); }

void InputDefinitionUpdater::resetRedCB() { PRESERVE_ACTION(resetRedCB, inputDefinition); }

CurveTemplate<double>* InputDefinitionUpdater::displaceRedCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceRedCB, inputDefinition, delete, return, double, newCurve);
}

const CurveTemplate<double>& InputDefinitionUpdater::getRedCB() const { return inputDefinition->getRedCB(); }

void InputDefinitionUpdater::replaceRedCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceRedCB, inputDefinition, , , double, newCurve);
}

CurveTemplate<double>* InputDefinitionUpdater::displaceGreenCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceGreenCB, inputDefinition, delete, return, double, newCurve);
}

void InputDefinitionUpdater::resetGreenCB() { PRESERVE_ACTION(resetGreenCB, inputDefinition); }

const CurveTemplate<double>& InputDefinitionUpdater::getGreenCB() const { return inputDefinition->getGreenCB(); }

void InputDefinitionUpdater::replaceGreenCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceGreenCB, inputDefinition, , , double, newCurve);
}

void InputDefinitionUpdater::resetBlueCB() { PRESERVE_ACTION(resetBlueCB, inputDefinition); }

void InputDefinitionUpdater::replaceBlueCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceBlueCB, inputDefinition, , , double, newCurve);
}

CurveTemplate<double>* InputDefinitionUpdater::displaceBlueCB(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceBlueCB, inputDefinition, delete, return, double, newCurve);
}

const CurveTemplate<double>& InputDefinitionUpdater::getBlueCB() const { return inputDefinition->getBlueCB(); }

void InputDefinitionUpdater::resetExposureValue() { PRESERVE_ACTION(resetExposureValue, inputDefinition); }

void InputDefinitionUpdater::replaceExposureValue(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceExposureValue, inputDefinition, , , double, newCurve);
}

CurveTemplate<double>* InputDefinitionUpdater::displaceExposureValue(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceExposureValue, inputDefinition, delete, return, double, newCurve);
}

const CurveTemplate<double>& InputDefinitionUpdater::getExposureValue() const {
  return inputDefinition->getExposureValue();
}

CurveTemplate<GeometryDefinition>* InputDefinitionUpdater::displaceGeometries(
    CurveTemplate<GeometryDefinition>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceGeometries, inputDefinition, delete, return, GeometryDefinition, newCurve);
}

void InputDefinitionUpdater::replaceGeometries(CurveTemplate<GeometryDefinition>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceGeometries, inputDefinition, , , GeometryDefinition, newCurve);
}

const CurveTemplate<GeometryDefinition>& InputDefinitionUpdater::getGeometries() const {
  return inputDefinition->getGeometries();
}

void InputDefinitionUpdater::resetGeometries(const double HFOV) {
  PRESERVE_ACTION(resetGeometries, inputDefinition, HFOV);
}

const Ptv::Value* InputDefinitionUpdater::getPreprocessors() const { return inputDefinition->getPreprocessors(); }

void InputDefinitionUpdater::setIsEnabled(bool state) { PRESERVE_ACTION(setIsEnabled, inputDefinition, state); }

bool InputDefinitionUpdater::getIsEnabled() const { return inputDefinition->getIsEnabled(); }

bool InputDefinitionUpdater::getIsVideoEnabled() const { return inputDefinition->getIsVideoEnabled(); }

bool InputDefinitionUpdater::getIsAudioEnabled() const { return inputDefinition->getIsAudioEnabled(); }

void InputDefinitionUpdater::setUseMeterDistortion(bool meter) {
  PRESERVE_ACTION(setUseMeterDistortion, inputDefinition, meter);
}

InputDefinition::PhotoResponse InputDefinitionUpdater::getPhotoResponse() const {
  return inputDefinition->getPhotoResponse();
}

double InputDefinitionUpdater::getEmorA() const { return inputDefinition->getEmorA(); }

double InputDefinitionUpdater::getEmorB() const { return inputDefinition->getEmorB(); }

double InputDefinitionUpdater::getEmorC() const { return inputDefinition->getEmorC(); }

double InputDefinitionUpdater::getEmorD() const { return inputDefinition->getEmorD(); }

double InputDefinitionUpdater::getEmorE() const { return inputDefinition->getEmorE(); }

double InputDefinitionUpdater::getGamma() const { return inputDefinition->getGamma(); }

void InputDefinitionUpdater::setEmorA(double emorA) { PRESERVE_ACTION(setEmorA, inputDefinition, emorA); }

void InputDefinitionUpdater::setEmorB(double emorB) { PRESERVE_ACTION(setEmorB, inputDefinition, emorB); }

void InputDefinitionUpdater::setEmorC(double emorC) { PRESERVE_ACTION(setEmorC, inputDefinition, emorC); }

void InputDefinitionUpdater::setEmorD(double emorD) { PRESERVE_ACTION(setEmorD, inputDefinition, emorD); }

void InputDefinitionUpdater::setEmorE(double emorE) { PRESERVE_ACTION(setEmorE, inputDefinition, emorE); }

void InputDefinitionUpdater::setEmorPhotoResponse(double emorA, double emorB, double emorC, double emorD,
                                                  double emorE) {
  PRESERVE_ACTION(setEmorPhotoResponse, inputDefinition, emorA, emorB, emorC, emorD, emorE);
}

void InputDefinitionUpdater::resetPhotoResponse() { PRESERVE_ACTION(resetPhotoResponse, inputDefinition); }

void InputDefinitionUpdater::setGamma(double gamma) { PRESERVE_ACTION(setGamma, inputDefinition, gamma); }

double InputDefinitionUpdater::getVignettingCoeff0() const { return inputDefinition->getVignettingCoeff0(); }

double InputDefinitionUpdater::getVignettingCoeff1() const { return inputDefinition->getVignettingCoeff1(); }

double InputDefinitionUpdater::getVignettingCoeff2() const { return inputDefinition->getVignettingCoeff2(); }

double InputDefinitionUpdater::getVignettingCoeff3() const { return inputDefinition->getVignettingCoeff3(); }

double InputDefinitionUpdater::getVignettingCenterX() const { return inputDefinition->getVignettingCenterX(); }

double InputDefinitionUpdater::getVignettingCenterY() const { return inputDefinition->getVignettingCenterY(); }

void InputDefinitionUpdater::setVignettingCoeff0(double vignettingCoeff0) {
  PRESERVE_ACTION(setVignettingCoeff0, inputDefinition, vignettingCoeff0);
}

void InputDefinitionUpdater::setVignettingCoeff1(double vignettingCoeff1) {
  PRESERVE_ACTION(setVignettingCoeff1, inputDefinition, vignettingCoeff1);
}

void InputDefinitionUpdater::setVignettingCoeff2(double vignettingCoeff2) {
  PRESERVE_ACTION(setVignettingCoeff2, inputDefinition, vignettingCoeff2);
}

void InputDefinitionUpdater::setVignettingCoeff3(double vignettingCoeff3) {
  PRESERVE_ACTION(setVignettingCoeff3, inputDefinition, vignettingCoeff3);
}

void InputDefinitionUpdater::setVignettingCenterX(double vignettingCenterX) {
  PRESERVE_ACTION(setVignettingCenterX, inputDefinition, vignettingCenterX);
}

void InputDefinitionUpdater::setVignettingCenterY(double vignettingCenterY) {
  PRESERVE_ACTION(setVignettingCenterY, inputDefinition, vignettingCenterY);
}

void InputDefinitionUpdater::setRadialVignetting(double vignettingCoeff0, double vignettingCoeff1,
                                                 double vignettingCoeff2, double vignettingCoeff3,
                                                 double vignettingCenterX, double vignettingCenterY) {
  PRESERVE_ACTION(setRadialVignetting, inputDefinition, vignettingCoeff0, vignettingCoeff1, vignettingCoeff2,
                  vignettingCoeff3, vignettingCenterX, vignettingCenterY);
}

void InputDefinitionUpdater::resetVignetting() { PRESERVE_ACTION(resetVignetting, inputDefinition); }

double InputDefinitionUpdater::getInputCenterX() const { return inputDefinition->getInputCenterX(); }

double InputDefinitionUpdater::getInputCenterY() const { return inputDefinition->getInputCenterY(); }

double InputDefinitionUpdater::getCenterX(const GeometryDefinition& geometry) const {
  return inputDefinition->getCenterX(geometry);
}

double InputDefinitionUpdater::getCenterY(const GeometryDefinition& geometry) const {
  return inputDefinition->getCenterY(geometry);
}

int64_t InputDefinitionUpdater::getCropLeft() const { return inputDefinition->getCropLeft(); }

int64_t InputDefinitionUpdater::getCropRight() const { return inputDefinition->getCropRight(); }

int64_t InputDefinitionUpdater::getCropTop() const { return inputDefinition->getCropTop(); }

int64_t InputDefinitionUpdater::getCropBottom() const { return inputDefinition->getCropBottom(); }

void InputDefinitionUpdater::setCropLeft(int64_t left) { PRESERVE_ACTION(setCropLeft, inputDefinition, left); }

void InputDefinitionUpdater::setCropRight(int64_t right) { PRESERVE_ACTION(setCropRight, inputDefinition, right); }

void InputDefinitionUpdater::setCropTop(int64_t top) { PRESERVE_ACTION(setCropTop, inputDefinition, top); }

void InputDefinitionUpdater::setCropBottom(int64_t bottom) { PRESERVE_ACTION(setCropBottom, inputDefinition, bottom); }

void InputDefinitionUpdater::setCrop(int64_t left, int64_t right, int64_t top, int64_t bottom) {
  PRESERVE_ACTION(setCrop, inputDefinition, left, right, top, bottom);
}

}  // namespace Core
}  // namespace VideoStitch
