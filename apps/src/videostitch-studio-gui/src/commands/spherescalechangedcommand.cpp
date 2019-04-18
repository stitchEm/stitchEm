// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "spherescalechangedcommand.hpp"

#include "widgets/outputconfigurationwidget.hpp"

#include <QCoreApplication>

SphereScaleChangedCommand::SphereScaleChangedCommand(double oldSphereScale, double newSphereScale,
                                                     OutputConfigurationWidget* outputConfigurationWidget)
    : QUndoCommand(),
      oldSphereScale(oldSphereScale),
      newSphereScale(newSphereScale),
      outputConfigurationWidget(outputConfigurationWidget) {
  setText(QCoreApplication::translate("Undo command", "Sphere scale changed"));
}

void SphereScaleChangedCommand::undo() { outputConfigurationWidget->changeSphereScale(oldSphereScale); }

void SphereScaleChangedCommand::redo() { outputConfigurationWidget->changeSphereScale(newSphereScale); }
