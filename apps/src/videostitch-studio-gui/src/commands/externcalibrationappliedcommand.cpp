// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "externcalibrationappliedcommand.hpp"

#include "videostitcher/postprodprojectdefinition.hpp"

#include <QCoreApplication>

ExternCalibrationAppliedCommand::ExternCalibrationAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                                                 VideoStitch::Core::PanoDefinition* newPanoDef,
                                                                 PostProdProjectDefinition* project)
    : QObject(), QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myProject(project) {
  setText(QCoreApplication::translate("Undo command", "External calibration applied"));
}

ExternCalibrationAppliedCommand::~ExternCalibrationAppliedCommand() {
  delete myOldPanoDef;
  delete myNewPanoDef;
}

void ExternCalibrationAppliedCommand::undo() { emit reqApplyCalibration(myOldPanoDef->clone()); }

void ExternCalibrationAppliedCommand::redo() { emit reqApplyCalibration(myNewPanoDef->clone()); }
