// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panochangedcommand.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

PanoChangedCommand::PanoChangedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                       VideoStitch::Core::PanoDefinition* newPanoDef, QString text)
    : QUndoCommand(text), myOldPanoDef(oldPanoDef), myNewPanoDef(newPanoDef) {}

PanoChangedCommand::~PanoChangedCommand() {}

void PanoChangedCommand::undo() {
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  QMetaObject::invokeMethod(stitcherController, "changePano", Qt::AutoConnection,
                            Q_ARG(VideoStitch::Core::PanoDefinition*, myOldPanoDef->clone()));
}

void PanoChangedCommand::redo() {
  StitcherController* stitcherController = GlobalController::getInstance().getController();
  QMetaObject::invokeMethod(stitcherController, "changePano", Qt::AutoConnection,
                            Q_ARG(VideoStitch::Core::PanoDefinition*, myNewPanoDef->clone()));
}
