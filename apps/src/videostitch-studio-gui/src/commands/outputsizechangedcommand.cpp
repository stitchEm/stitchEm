// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputsizechangedcommand.hpp"
#include "centralwidget/processtab/outputfileprocess.hpp"
#include <QCoreApplication>

OutputSizeChangedCommand::OutputSizeChangedCommand(unsigned oldWidth, unsigned oldHeight, unsigned newWidth,
                                                   unsigned newHeight, OutputFileProcess *processwidget)
    : QUndoCommand(),
      myOldWidth(oldWidth),
      myOldHeight(oldHeight),
      myNewWidth(newWidth),
      myNewHeight(newHeight),
      myProcessWidget(processwidget) {
  setText(
      QCoreApplication::translate("Undo command", "Output size changed to %0 x %1").arg(myNewWidth).arg(myNewHeight));
}

void OutputSizeChangedCommand::undo() { myProcessWidget->setPanoramaSize(myOldWidth, myOldHeight); }

void OutputSizeChangedCommand::redo() { myProcessWidget->setPanoramaSize(myNewWidth, myNewHeight); }

bool OutputSizeChangedCommand::mergeWith(const QUndoCommand *otherCommand) {
  const OutputSizeChangedCommand *command = static_cast<const OutputSizeChangedCommand *>(otherCommand);
  myNewWidth = command->myNewWidth;
  myNewHeight = command->myNewHeight;
  setText(
      QCoreApplication::translate("Undo command", "Output size changed to %0 x %1").arg(myNewWidth).arg(myNewHeight));
  return true;
}
