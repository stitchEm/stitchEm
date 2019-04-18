// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SYNCHRONIZATIONOFFSETSCHANGEDCOMMAND_H
#define SYNCHRONIZATIONOFFSETSCHANGEDCOMMAND_H

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "widgets/synchronizationwidget.hpp"

class SynchronizationOffsetsChangedCommand : public QUndoCommand {
 public:
  SynchronizationOffsetsChangedCommand(QVector<int> newValues, QVector<int> oldValues, QVector<bool> newChecked,
                                       QVector<bool> oldChecked, int index,
                                       SynchronizationWidget* synchronizationWidget);

  int id() const { return UndoCommandId::SynchronizationOffsets; }
  void undo();
  void redo();
  bool mergeWith(const QUndoCommand* other);

 private:
  QVector<int> myPreviousValues;
  QVector<int> myNewValues;
  QVector<bool> myNewChecked;
  QVector<bool> myOldChecked;
  int myIndex;
  SynchronizationWidget* mySynchronizationWidget;
};

#endif  // SYNCHRONIZATIONOFFSETSCHANGEDCOMMAND_H
