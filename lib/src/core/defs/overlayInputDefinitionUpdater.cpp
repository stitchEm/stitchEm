// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/overlayInputDefinitionUpdater.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Core {

OverlayInputDefinitionUpdater::OverlayInputDefinitionUpdater(const OverlayInputDefinition& overlayInputDefinition)
    : overlayInputDefinition(overlayInputDefinition.clone()) {}

OverlayInputDefinitionUpdater::OverlayInputDefinitionUpdater(OverlayInputDefinition* overlayInputDefinition)
    : overlayInputDefinition(overlayInputDefinition) {}

OverlayInputDefinition* OverlayInputDefinitionUpdater::clone() const { return overlayInputDefinition->clone(); }

Ptv::Value* OverlayInputDefinitionUpdater::serialize() const { return overlayInputDefinition->serialize(); }

bool OverlayInputDefinitionUpdater::operator==(const OverlayInputDefinition& other) const {
  return overlayInputDefinition->operator==(other);
}

bool OverlayInputDefinitionUpdater::validate(std::ostream& os) const { return overlayInputDefinition->validate(os); }

const Ptv::Value& OverlayInputDefinitionUpdater::getReaderConfig() const {
  return overlayInputDefinition->getReaderConfig();
}

void OverlayInputDefinitionUpdater::setReaderConfig(Ptv::Value* config) {
  PRESERVE_ACTION_CLONEABLE(setReaderConfig, overlayInputDefinition, , , Ptv::Value, config);
}

frameid_t OverlayInputDefinitionUpdater::getFrameOffset() const { return overlayInputDefinition->getFrameOffset(); }

void OverlayInputDefinitionUpdater::setFrameOffset(int fo) {
  PRESERVE_ACTION(setFrameOffset, overlayInputDefinition, fo);
}

int64_t OverlayInputDefinitionUpdater::getWidth() const { return overlayInputDefinition->getWidth(); }

void OverlayInputDefinitionUpdater::setWidth(int64_t int64) {
  PRESERVE_ACTION(setWidth, overlayInputDefinition, int64);
}

int64_t OverlayInputDefinitionUpdater::getHeight() const { return overlayInputDefinition->getHeight(); }

void OverlayInputDefinitionUpdater::setHeight(int64_t int64) {
  PRESERVE_ACTION(setHeight, overlayInputDefinition, int64);
}

void OverlayInputDefinitionUpdater::setFilename(const std::string& fileName) {
  PRESERVE_ACTION(setFilename, overlayInputDefinition, fileName);
}

std::string OverlayInputDefinitionUpdater::getDisplayName() const { return overlayInputDefinition->getDisplayName(); }

bool OverlayInputDefinitionUpdater::getGlobalOrientationApplied() const {
  return overlayInputDefinition->getGlobalOrientationApplied();
}

void OverlayInputDefinitionUpdater::setGlobalOrientationApplied(const bool status) {
  PRESERVE_ACTION(setGlobalOrientationApplied, overlayInputDefinition, status);
}

const CurveTemplate<double>& OverlayInputDefinitionUpdater::getScaleCurve() const {
  return overlayInputDefinition->getScaleCurve();
}

CurveTemplate<double>* OverlayInputDefinitionUpdater::displaceScaleCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceScaleCurve, overlayInputDefinition, delete, return, double, newCurve);
}

void OverlayInputDefinitionUpdater::resetScaleCurve() { PRESERVE_ACTION(resetScaleCurve, overlayInputDefinition); }

void OverlayInputDefinitionUpdater::replaceScaleCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceScaleCurve, overlayInputDefinition, , , double, newCurve);
}

const CurveTemplate<double>& OverlayInputDefinitionUpdater::getTransXCurve() const {
  return overlayInputDefinition->getTransXCurve();
}

CurveTemplate<double>* OverlayInputDefinitionUpdater::displaceTransXCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceTransXCurve, overlayInputDefinition, delete, return, double, newCurve);
}

void OverlayInputDefinitionUpdater::resetTransXCurve() { PRESERVE_ACTION(resetTransXCurve, overlayInputDefinition); }

void OverlayInputDefinitionUpdater::replaceTransXCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceTransXCurve, overlayInputDefinition, , , double, newCurve);
}

const CurveTemplate<double>& OverlayInputDefinitionUpdater::getTransYCurve() const {
  return overlayInputDefinition->getTransYCurve();
}

CurveTemplate<double>* OverlayInputDefinitionUpdater::displaceTransYCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceTransYCurve, overlayInputDefinition, delete, return, double, newCurve);
}

void OverlayInputDefinitionUpdater::resetTransYCurve() { PRESERVE_ACTION(resetTransYCurve, overlayInputDefinition); }

void OverlayInputDefinitionUpdater::replaceTransYCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceTransYCurve, overlayInputDefinition, , , double, newCurve);
}

const CurveTemplate<double>& OverlayInputDefinitionUpdater::getTransZCurve() const {
  return overlayInputDefinition->getTransZCurve();
}

CurveTemplate<double>* OverlayInputDefinitionUpdater::displaceTransZCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceTransZCurve, overlayInputDefinition, delete, return, double, newCurve);
}

void OverlayInputDefinitionUpdater::resetTransZCurve() { PRESERVE_ACTION(resetTransZCurve, overlayInputDefinition); }

void OverlayInputDefinitionUpdater::replaceTransZCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceTransZCurve, overlayInputDefinition, , , double, newCurve);
}

const CurveTemplate<double>& OverlayInputDefinitionUpdater::getAlphaCurve() const {
  return overlayInputDefinition->getAlphaCurve();
}

CurveTemplate<double>* OverlayInputDefinitionUpdater::displaceAlphaCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceAlphaCurve, overlayInputDefinition, delete, return, double, newCurve);
}

void OverlayInputDefinitionUpdater::resetAlphaCurve() { PRESERVE_ACTION(resetAlphaCurve, overlayInputDefinition); }

void OverlayInputDefinitionUpdater::replaceAlphaCurve(CurveTemplate<double>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceAlphaCurve, overlayInputDefinition, , , double, newCurve);
}

const CurveTemplate<Quaternion<double>>& OverlayInputDefinitionUpdater::getRotationCurve() const {
  return overlayInputDefinition->getRotationCurve();
}

CurveTemplate<Quaternion<double>>* OverlayInputDefinitionUpdater::displaceRotationCurve(
    CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(displaceRotationCurve, overlayInputDefinition, delete, return, Quaternion<double>, newCurve);
}

void OverlayInputDefinitionUpdater::resetRotationCurve() {
  PRESERVE_ACTION(resetRotationCurve, overlayInputDefinition);
}

void OverlayInputDefinitionUpdater::replaceRotationCurve(CurveTemplate<Quaternion<double>>* newCurve) {
  PRESERVE_ACTION_CURVE(replaceRotationCurve, overlayInputDefinition, , , Quaternion<double>, newCurve);
}
}  // namespace Core
}  // namespace VideoStitch
