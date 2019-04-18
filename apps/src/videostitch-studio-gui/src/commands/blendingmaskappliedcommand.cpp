// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "blendingmaskappliedcommand.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include <QCoreApplication>

BlendingMaskAppliedCommand::BlendingMaskAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                                       VideoStitch::Core::PanoDefinition* newPanoDef,
                                                       BlendingMaskWidget* blendingMaskWidget)
    : QUndoCommand(), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef), myBlendingMaskWidget(blendingMaskWidget) {
  setText(QCoreApplication::translate("Undo command", "Blending mask applied"));
}

BlendingMaskAppliedCommand::~BlendingMaskAppliedCommand() {
  delete myOldPanoDef;
  delete myNewPanoDef;
}

void BlendingMaskAppliedCommand::undo() { emit myBlendingMaskWidget->reqApplyBlendingMask(myOldPanoDef->clone()); }

void BlendingMaskAppliedCommand::redo() { emit myBlendingMaskWidget->reqApplyBlendingMask(myNewPanoDef->clone()); }
