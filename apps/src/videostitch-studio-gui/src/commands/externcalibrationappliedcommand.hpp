// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EXTERNCALIBRATIONAPPLIEDCOMMAND_HPP
#define EXTERNCALIBRATIONAPPLIEDCOMMAND_HPP

#include "undoHelpers.hpp"

#include <QUndoCommand>

namespace VideoStitch {
namespace Core {
class PanoDefinition;
}
}  // namespace VideoStitch
class PostProdProjectDefinition;

class ExternCalibrationAppliedCommand : public QObject, public QUndoCommand {
  Q_OBJECT

 public:
  ExternCalibrationAppliedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef,
                                  VideoStitch::Core::PanoDefinition* newPanoDef, PostProdProjectDefinition* project);
  ~ExternCalibrationAppliedCommand();

  int id() const { return UndoCommandId::ExternCalibrationApplied; }
  void undo();
  void redo();

 signals:
  void reqApplyCalibration(VideoStitch::Core::PanoDefinition* panoDef);

 private:
  VideoStitch::Core::PanoDefinition* myOldPanoDef;
  VideoStitch::Core::PanoDefinition* myNewPanoDef;
  PostProdProjectDefinition* myProject;
};

#endif  // EXTERNCALIBRATIONAPPLIEDCOMMAND_HPP
