// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BLENDERCHANGEDCOMMAND_HPP
#define BLENDERCHANGEDCOMMAND_HPP

#include "undoHelpers.hpp"

#include <QUndoCommand>

class OutputConfigurationWidget;

class BlenderChangedCommand : public QUndoCommand {
 public:
  BlenderChangedCommand(QString oldMerger, int oldFeather, QString newMerger, int newFeather,
                        OutputConfigurationWidget* outputConfigurationWidget);

  int id() const { return UndoCommandId::BlenderChanged; }
  void undo();
  void redo();

 private:
  QString oldMerger;
  int oldFeather;
  QString newMerger;
  int newFeather;
  OutputConfigurationWidget* outputConfigurationWidget;
};

#endif  // BLENDERCHANGEDCOMMAND_HPP
