// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "undoHelpers.hpp"

#include "libvideostitch-base/projection.hpp"

#include <QUndoCommand>

class OutputConfigurationWidget;

class ProjectionChangedCommand : public QUndoCommand {
 public:
  ProjectionChangedCommand(VideoStitch::Projection oldProjection, double oldFov, VideoStitch::Projection newProjection,
                           double newFov, OutputConfigurationWidget* outputConfigurationWidget);

  int id() const { return UndoCommandId::ProjectionChanged; }
  void undo();
  void redo();

 private:
  VideoStitch::Projection myOldProjection;
  VideoStitch::Projection myNewProjection;
  double myOldFov;
  double myNewFov;
  OutputConfigurationWidget* outputConfigurationWidget;
};
