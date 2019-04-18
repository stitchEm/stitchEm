// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stitchercontrollerprogressreporter.hpp"

#include "stitchercontroller.hpp"

StitcherControllerProgressReporter::StitcherControllerProgressReporter(StitcherController* stitcherController)
    : controller(stitcherController) {
  Q_ASSERT(controller != nullptr);
  controller->actionStarted();
}

StitcherControllerProgressReporter::~StitcherControllerProgressReporter() {
  if (!finished) {
    controller->actionCancelled();
  }
}

void StitcherControllerProgressReporter::setProgress(int progress) { controller->actionStep(progress); }

void StitcherControllerProgressReporter::finishProgress() {
  finished = true;
  controller->actionFinished();
}
