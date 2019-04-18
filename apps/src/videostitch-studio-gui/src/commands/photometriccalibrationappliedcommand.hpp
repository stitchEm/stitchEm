// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "undoHelpers.hpp"

#include <QUndoCommand>

namespace VideoStitch {
namespace Core {
class PanoDefinition;
}
}  // namespace VideoStitch
class ExposureWidget;

class PhotometricCalibrationAppliedCommand : public QUndoCommand {
 public:
  /* The command takes the ownership of oldPanoDef and newPanoDef */
  PhotometricCalibrationAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                       VideoStitch::Core::PanoDefinition* newPanoDef, ExposureWidget* exposureWidget);
  ~PhotometricCalibrationAppliedCommand();

  int id() const { return UndoCommandId::PhotometricCalibrationApplied; }
  void undo();
  void redo();

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  ExposureWidget* myExposureWidget;
};
