// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "orientationchangedcommand.hpp"
#include <QCoreApplication>

OrientationChangedCommand::OrientationChangedCommand(int frame, VideoStitch::Quaternion<double> oldOrientation,
                                                     VideoStitch::Quaternion<double> newOrientation,
                                                     VideoStitch::Core::QuaternionCurve* curve)
    : QObject(),
      QUndoCommand(),
      myFrame(frame),
      myOldOrientation(oldOrientation),
      myNewOrientation(newOrientation),
      myCurve(curve) {
  setText(QCoreApplication::translate("Undo command", "Orientation edited"));
}

OrientationChangedCommand::~OrientationChangedCommand() { delete myCurve; }

void OrientationChangedCommand::undo() { emit reqFinishOrientation(myFrame, myOldOrientation, myCurve); }

void OrientationChangedCommand::redo() { emit reqFinishOrientation(myFrame, myNewOrientation, myCurve); }
