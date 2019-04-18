// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "workingareachangedcommand.hpp"
#include <QCoreApplication>

WorkingAreaChangedCommand::WorkingAreaChangedCommand(qint64 oldFirst, qint64 oldLast, qint64 newFirst, qint64 newLast,
                                                     SeekBar* seekbar)
    : QUndoCommand(),
      myOldFirst(oldFirst),
      myOldLast(oldLast),
      myNewFirst(newFirst),
      myNewLast(newLast),
      mySeekBar(seekbar) {
  setText(QCoreApplication::translate("Undo command", "Working area changed"));
}

void WorkingAreaChangedCommand::undo() { mySeekBar->setWorkingArea(myOldFirst, myOldLast); }

void WorkingAreaChangedCommand::redo() { mySeekBar->setWorkingArea(myNewFirst, myNewLast); }

bool WorkingAreaChangedCommand::mergeWith(const QUndoCommand* otherCommand) {
  const WorkingAreaChangedCommand* command = static_cast<const WorkingAreaChangedCommand*>(otherCommand);
  myNewFirst = command->myNewFirst;
  myNewLast = command->myNewLast;
  return true;
}
