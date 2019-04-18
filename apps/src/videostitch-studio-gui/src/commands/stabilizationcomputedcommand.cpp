// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stabilizationcomputedcommand.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include <QCoreApplication>

StabilizationComputedCommand::StabilizationComputedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                                           VideoStitch::Core::PanoDefinition* newPanoDef,
                                                           StabilizationWidget* stabilizationWidget)
    : QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myStabilizationWidget(stabilizationWidget) {
  setText(QCoreApplication::translate("Undo command", "Compute stabilization"));
}

void StabilizationComputedCommand::undo() {
  emit myStabilizationWidget->reqApplyStabilization(myOldPanoDef->clone());
  ;
}

void StabilizationComputedCommand::redo() {
  emit myStabilizationWidget->reqApplyStabilization(myNewPanoDef->clone());
  ;
}
