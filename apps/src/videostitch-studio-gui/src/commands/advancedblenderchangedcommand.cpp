// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "advancedblenderchangedcommand.hpp"

#include "widgets/advancedblendingwidget.hpp"

#include <QCoreApplication>

AdvancedBlenderChangedCommand::AdvancedBlenderChangedCommand(const QString& oldFlow, const QString oldWarper,
                                                             const QString& newFlow, const QString newWarper,
                                                             AdvancedBlendingWidget* advancedBlendingWidget)
    : QUndoCommand(),
      oldFlow(oldFlow),
      oldWarper(oldWarper),
      newFlow(newFlow),
      newWarper(newWarper),
      advancedBlendingWidget(advancedBlendingWidget) {
  setText(QCoreApplication::translate("Undo command", "Advanced blender changed"));
}

void AdvancedBlenderChangedCommand::undo() { advancedBlendingWidget->changeBlender(oldFlow, oldWarper); }

void AdvancedBlenderChangedCommand::redo() { advancedBlendingWidget->changeBlender(newFlow, newWarper); }
