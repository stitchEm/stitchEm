// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef STABILIZATIONCOMPUTEDCOMMAND_HPP
#define STABILIZATIONCOMPUTEDCOMMAND_HPP

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "../widgets/stabilizationwidget.hpp"

class StabilizationComputedCommand : public QUndoCommand {
 public:
  StabilizationComputedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                               VideoStitch::Core::PanoDefinition* newPanoDef, StabilizationWidget* stabilizationwidget);

  int id() const { return UndoCommandId::StabilizationComputed; }
  void undo();
  void redo();

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  StabilizationWidget* myStabilizationWidget;
};

#endif  // STABILIZATIONCOMPUTEDCOMMAND_HPP
