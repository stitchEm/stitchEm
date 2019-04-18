// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "projectionchangedcommand.hpp"

#include "widgets/outputconfigurationwidget.hpp"

#include <QCoreApplication>

ProjectionChangedCommand::ProjectionChangedCommand(VideoStitch::Projection oldProjection, double oldFov,
                                                   VideoStitch::Projection newProjection, double newFov,
                                                   OutputConfigurationWidget* outputConfigurationWidget)
    : QUndoCommand(),
      myOldProjection(oldProjection),
      myNewProjection(newProjection),
      myOldFov(oldFov),
      myNewFov(newFov),
      outputConfigurationWidget(outputConfigurationWidget) {
  setText(QCoreApplication::translate("Undo command", "Projection and H.Fov changed"));
}

void ProjectionChangedCommand::undo() { outputConfigurationWidget->changeProjectionAndFov(myOldProjection, myOldFov); }

void ProjectionChangedCommand::redo() { outputConfigurationWidget->changeProjectionAndFov(myNewProjection, myNewFov); }
