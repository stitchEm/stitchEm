// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CALIBRATIONAPPLIEDCOMMAND_HPP
#define CALIBRATIONAPPLIEDCOMMAND_HPP

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "widgets/calibrationwidget.hpp"

class CalibrationAppliedCommand : public QUndoCommand {
 public:
  /* The command takes the ownership of oldPanoDef and newPanoDef */
  CalibrationAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                            VideoStitch::Core::PanoDefinition* newPanoDef, CalibrationWidget* calibrationWidget);
  ~CalibrationAppliedCommand();

  int id() const { return UndoCommandId::CalibrationApplied; }
  void undo();
  void redo();

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  CalibrationWidget* myCalibrationWidget;
};

#endif  // CALIBRATIONAPPLIEDCOMMAND_HPP
