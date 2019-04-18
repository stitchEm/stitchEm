// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

class StitcherController;

class VS_GUI_EXPORT StitcherControllerProgressReporter {
 public:
  explicit StitcherControllerProgressReporter(StitcherController* stitcherController);
  ~StitcherControllerProgressReporter();

  void setProgress(int progress);
  void finishProgress();

 private:
  StitcherController* controller = nullptr;
  bool finished = false;
};
