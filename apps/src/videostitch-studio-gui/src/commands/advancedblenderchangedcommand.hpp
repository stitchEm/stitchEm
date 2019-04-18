// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ADVANCEDBLENDERCHANGEDCOMMAND_HPP
#define ADVANCEDBLENDERCHANGEDCOMMAND_HPP

#include "undoHelpers.hpp"

#include <QUndoCommand>

class AdvancedBlendingWidget;

class AdvancedBlenderChangedCommand : public QUndoCommand {
 public:
  AdvancedBlenderChangedCommand(const QString& oldFlow, const QString oldWarper, const QString& newFlow,
                                const QString newWarper, AdvancedBlendingWidget* advancedBlendingWidget);

  int id() const { return UndoCommandId::AdvancedBlenderChanged; }
  void undo();
  void redo();

 private:
  QString oldFlow;
  QString oldWarper;
  QString newFlow;
  QString newWarper;
  AdvancedBlendingWidget* advancedBlendingWidget;
};

#endif  // ADVANCEDBLENDERCHANGEDCOMMAND_HPP
