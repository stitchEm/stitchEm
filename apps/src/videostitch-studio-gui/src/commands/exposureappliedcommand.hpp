// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EXPOSUREAPPLIEDCOMMAND_HPP
#define EXPOSUREAPPLIEDCOMMAND_HPP

#include <QUndoCommand>
#include "undoHelpers.hpp"
#include "widgets/exposurewidget.hpp"

class ExposureAppliedCommand : public QUndoCommand {
 public:
  /* The command takes the ownership of oldPanoDef and newPanoDef */
  ExposureAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef, VideoStitch::Core::PanoDefinition* newPanoDef,
                         ExposureWidget* exposureWidget);
  ~ExposureAppliedCommand();

  int id() const { return UndoCommandId::ExposureApplied; }
  void undo();
  void redo();

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  ExposureWidget* myExposureWidget;
};

#endif  // EXPOSUREAPPLIEDCOMMAND_HPP
