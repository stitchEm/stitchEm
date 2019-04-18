// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef UNDOHELPERS_HPP
#define UNDOHELPERS_HPP

enum UndoCommandId {
  SynchronizationOffsets = 0,
  CalibrationApplied = 1,
  ExternCalibrationApplied = 2,
  OutputSizeChanged = 3,
  BlenderChanged = 4,
  StabilizationComputed = 5,
  WorkingAreaChanged = 6,
  ExposureApplied = 7,
  OrientationChanged = 8,
  SeekingFrame = 9,
  TimeEdited = 10,
  ProjectionChanged = 11,
  HFovEdited = 12,
  BlendingMaskApplied = 13,
  PhotometricCalibrationApplied = 14,
  AdvancedBlenderChanged = 15,
  SphereScaleChanged = 16
};

#endif  // UNDOHELPERS_HPP
