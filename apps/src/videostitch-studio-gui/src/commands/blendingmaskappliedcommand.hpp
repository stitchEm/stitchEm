// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "widgets/blendingmaskwidget.hpp"

class BlendingMaskAppliedCommand : public QUndoCommand {
 public:
  /* The command takes the ownership of oldPanoDef and newPanoDef */
  BlendingMaskAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                             VideoStitch::Core::PanoDefinition* newPanoDef, BlendingMaskWidget* blendingMaskWidget);
  ~BlendingMaskAppliedCommand();

  int id() const { return UndoCommandId::BlendingMaskApplied; }
  void undo();
  void redo();

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  BlendingMaskWidget* myBlendingMaskWidget;
};
