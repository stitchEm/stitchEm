// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureappliedcommand.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include <QCoreApplication>

ExposureAppliedCommand::ExposureAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                               VideoStitch::Core::PanoDefinition* newPanoDef,
                                               ExposureWidget* exposureWidget)
    : QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myExposureWidget(exposureWidget) {
  setText(QCoreApplication::translate("Undo command", "Exposure applied"));
}

ExposureAppliedCommand::~ExposureAppliedCommand() {
  delete myOldPanoDef;
  delete myNewPanoDef;
}

void ExposureAppliedCommand::undo() { emit myExposureWidget->reqApplyExposure(myOldPanoDef->clone()); }

void ExposureAppliedCommand::redo() { emit myExposureWidget->reqApplyExposure(myNewPanoDef->clone()); }
