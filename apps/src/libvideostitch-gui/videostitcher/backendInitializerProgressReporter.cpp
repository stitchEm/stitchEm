// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backendInitializerProgressReporter.hpp"
#include "stitchercontroller.hpp"

BackendInitializerProgressReporter::BackendInitializerProgressReporter(StitcherController* stitcherController)
    : controller(stitcherController), cancel(false) {}

bool BackendInitializerProgressReporter::notify(const std::string& message, double percent) {
  controller->forwardBackendProgress(QString::fromStdString(message), percent);
  return cancel;
}

void BackendInitializerProgressReporter::tryToCancel() { cancel = true; }

void BackendInitializerProgressReporter::reset() { cancel = false; }
