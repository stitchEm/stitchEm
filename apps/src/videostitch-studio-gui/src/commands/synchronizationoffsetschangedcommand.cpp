// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "synchronizationoffsetschangedcommand.hpp"
#include <QCoreApplication>

SynchronizationOffsetsChangedCommand::SynchronizationOffsetsChangedCommand(QVector<int> newValues,
                                                                           QVector<int> oldValues,
                                                                           QVector<bool> newChecked,
                                                                           QVector<bool> oldChecked, int index,
                                                                           SynchronizationWidget *synchronizationWidget)
    : QUndoCommand(),
      myPreviousValues(oldValues),
      myNewValues(newValues),
      myNewChecked(newChecked),
      myOldChecked(oldChecked),
      myIndex(index),
      mySynchronizationWidget(synchronizationWidget) {
  setText(QCoreApplication::translate("Undo command", "Change synchronization offsets"));
}

void SynchronizationOffsetsChangedCommand::undo() {
  mySynchronizationWidget->changeAllValues(myPreviousValues, myOldChecked);
}

void SynchronizationOffsetsChangedCommand::redo() {
  mySynchronizationWidget->changeAllValues(myNewValues, myNewChecked);
}

bool SynchronizationOffsetsChangedCommand::mergeWith(const QUndoCommand *otherCommand) {
  const SynchronizationOffsetsChangedCommand *command =
      static_cast<const SynchronizationOffsetsChangedCommand *>(otherCommand);
  if ((myIndex == command->myIndex) && (command->myIndex != -1)) {
    myNewValues = command->myNewValues;
    myNewChecked = command->myNewChecked;
    return true;
  } else {
    return false;
  }
}
