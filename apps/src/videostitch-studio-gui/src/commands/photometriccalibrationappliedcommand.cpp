// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "photometriccalibrationappliedcommand.hpp"

#include "widgets/exposurewidget.hpp"

#include "libvideostitch/panoDef.hpp"

#include <QCoreApplication>

PhotometricCalibrationAppliedCommand::PhotometricCalibrationAppliedCommand(
    VideoStitch::Core::PanoDefinition* oldPanoDef, VideoStitch::Core::PanoDefinition* newPanoDef,
    ExposureWidget* exposureWidget)
    : QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myExposureWidget(exposureWidget) {
  setText(QCoreApplication::translate("Undo command", "Photometric calibration applied"));
}

PhotometricCalibrationAppliedCommand::~PhotometricCalibrationAppliedCommand() {
  delete myOldPanoDef;
  delete myNewPanoDef;
}

void PhotometricCalibrationAppliedCommand::undo() {
  emit myExposureWidget->reqApplyPhotometricCalibration(myOldPanoDef->clone());
}

void PhotometricCalibrationAppliedCommand::redo() {
  emit myExposureWidget->reqApplyPhotometricCalibration(myNewPanoDef->clone());
}
