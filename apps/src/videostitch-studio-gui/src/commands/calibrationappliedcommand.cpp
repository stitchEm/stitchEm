// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationappliedcommand.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include <QCoreApplication>

CalibrationAppliedCommand::CalibrationAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                                     VideoStitch::Core::PanoDefinition* newPanoDef,
                                                     CalibrationWidget* calibrationWidget)
    : QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myCalibrationWidget(calibrationWidget) {
  setText(QCoreApplication::translate("Undo command", "Calibration applied"));
}

CalibrationAppliedCommand::~CalibrationAppliedCommand() {
  delete myOldPanoDef;
  delete myNewPanoDef;
}

void CalibrationAppliedCommand::undo() { myCalibrationWidget->applyCalibration(myOldPanoDef->clone()); }

void CalibrationAppliedCommand::redo() { myCalibrationWidget->applyCalibration(myNewPanoDef->clone()); }
