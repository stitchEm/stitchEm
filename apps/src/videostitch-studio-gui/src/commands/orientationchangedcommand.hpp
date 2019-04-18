// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ORIENTATIONCHANGEDCOMMAND_HPP
#define ORIENTATIONCHANGEDCOMMAND_HPP

// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QUndoCommand>
#include "undoHelpers.hpp"

#include "libvideostitch/curves.hpp"
#include "libvideostitch/quaternion.hpp"

class OrientationChangedCommand : public QObject, public QUndoCommand {
  Q_OBJECT

 public:
  OrientationChangedCommand(int frame, VideoStitch::Quaternion<double> oldOrientation,
                            VideoStitch::Quaternion<double> newOrientation, VideoStitch::Core::QuaternionCurve* curve);
  ~OrientationChangedCommand();

  int id() const { return UndoCommandId::OrientationChanged; }
  void undo();
  void redo();

 signals:
  void reqFinishOrientation(int frame, VideoStitch::Quaternion<double> orientation,
                            VideoStitch::Core::QuaternionCurve* curve);

 private:
  int myFrame;
  VideoStitch::Quaternion<double> myOldOrientation;
  VideoStitch::Quaternion<double> myNewOrientation;
  VideoStitch::Core::QuaternionCurve* myCurve;
};

#endif  // ORIENTATIONCHANGEDCOMMAND_HPP
