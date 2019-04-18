// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QUndoCommand>

namespace VideoStitch {
namespace Core {
class PanoDefinition;
}
}  // namespace VideoStitch

class PanoChangedCommand : public QUndoCommand {
 public:
  /* The command takes the ownership of oldPanoDef and newPanoDef */
  PanoChangedCommand(VideoStitch::Core::PanoDefinition* oldPanoDef, VideoStitch::Core::PanoDefinition* newPanoDef,
                     QString text);
  ~PanoChangedCommand();

  virtual void undo();
  virtual void redo();

 private:
  QScopedPointer<VideoStitch::Core::PanoDefinition> myOldPanoDef;
  QScopedPointer<VideoStitch::Core::PanoDefinition> myNewPanoDef;
};
