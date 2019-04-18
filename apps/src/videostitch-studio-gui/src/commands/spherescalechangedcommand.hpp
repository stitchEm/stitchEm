// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "undoHelpers.hpp"

#include <QUndoCommand>

class OutputConfigurationWidget;

class SphereScaleChangedCommand : public QUndoCommand {
 public:
  SphereScaleChangedCommand(double oldSphereScale, double newSphereScale,
                            OutputConfigurationWidget* outputConfigurationWidget);

  int id() const { return UndoCommandId::SphereScaleChanged; }
  void undo();
  void redo();

 private:
  double oldSphereScale;
  double newSphereScale;
  OutputConfigurationWidget* outputConfigurationWidget;
};
