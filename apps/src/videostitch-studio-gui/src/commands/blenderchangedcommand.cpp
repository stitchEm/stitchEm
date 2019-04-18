// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "blenderchangedcommand.hpp"

#include "widgets/outputconfigurationwidget.hpp"

#include <QCoreApplication>

BlenderChangedCommand::BlenderChangedCommand(QString oldMerger, int oldFeather, QString newMerger, int newFeather,
                                             OutputConfigurationWidget* outputConfigurationWidget)
    : QUndoCommand(),
      oldMerger(oldMerger),
      oldFeather(oldFeather),
      newMerger(newMerger),
      newFeather(newFeather),
      outputConfigurationWidget(outputConfigurationWidget) {
  setText(QCoreApplication::translate("Undo command", "Blender changed"));
}

void BlenderChangedCommand::undo() { outputConfigurationWidget->changeBlender(oldMerger, oldFeather); }

void BlenderChangedCommand::redo() { outputConfigurationWidget->changeBlender(newMerger, newFeather); }
