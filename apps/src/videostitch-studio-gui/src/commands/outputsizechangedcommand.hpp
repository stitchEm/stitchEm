// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QUndoCommand>
#include "undoHelpers.hpp"

class OutputFileProcess;
class OutputSizeChangedCommand : public QUndoCommand {
 public:
  explicit OutputSizeChangedCommand(unsigned oldWidth, unsigned oldHeight, unsigned newWidth, unsigned newHeight,
                                    OutputFileProcess* processwidget);

  virtual int id() const override { return UndoCommandId::OutputSizeChanged; }
  virtual void undo() override;
  virtual void redo() override;
  virtual bool mergeWith(const QUndoCommand* other) override;

 private:
  unsigned myOldWidth;
  unsigned myOldHeight;
  unsigned myNewWidth;
  unsigned myNewHeight;
  OutputFileProcess* myProcessWidget;
};
