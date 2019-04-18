// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef WORKINGAREACHANGEDCOMMAND_HPP
#define WORKINGAREACHANGEDCOMMAND_HPP

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "widgets/seekbar.hpp"

class WorkingAreaChangedCommand : public QUndoCommand {
 public:
  WorkingAreaChangedCommand(qint64 oldFirst, qint64 oldLast, qint64 newFirst, qint64 newLast, SeekBar* seekbar);

  int id() const { return UndoCommandId::WorkingAreaChanged; }
  void undo();
  void redo();
  bool mergeWith(const QUndoCommand* otherCommand);

 private:
  qint64 myOldFirst;
  qint64 myOldLast;
  qint64 myNewFirst;
  qint64 myNewLast;
  SeekBar* mySeekBar;
};

#endif  // WORKINGAREACHANGEDCOMMAND_HPP
