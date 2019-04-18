// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationProgress.hpp"

#include <iostream>

namespace VideoStitch {
namespace Calibration {

const double CalibrationProgress::seek = 2.0;
const double CalibrationProgress::kpDetect = 1.0;
const double CalibrationProgress::kpMatch = 2.0;
const double CalibrationProgress::fovIterate = 0.5;
const double CalibrationProgress::deshuffle = 1.0;
const double CalibrationProgress::filter = 1.0;
const double CalibrationProgress::initGeometry = 0.1;
const double CalibrationProgress::optim = 1.0;
const double CalibrationProgress::optim_done = 0.001;

CalibrationProgress::CalibrationProgress(VideoStitch::Util::Algorithm::ProgressReporter* progress,
                                         const double totalUnits)
    : progress(progress), progressPercentage(0.0), unitPercentage(0.0), enabled(true) {
  if (totalUnits > 0) {
    unitPercentage = 100.0 / totalUnits;
  }
}

CalibrationProgress::CalibrationProgress(const CalibrationProgress& other)
    : progress(other.progress),
      progressPercentage(other.progressPercentage),
      unitPercentage(other.unitPercentage),
      enabled(other.enabled) {}

void CalibrationProgress::enable() {
  std::lock_guard<std::mutex> lock(progressLock);  // ensure thread-safety

  enabled = true;
}

void CalibrationProgress::disable() {
  std::lock_guard<std::mutex> lock(progressLock);  // ensure thread-safety

  enabled = false;
}

Status CalibrationProgress::add(double units, const std::string& msg) {
  std::lock_guard<std::mutex> lock(progressLock);  // ensure thread-safety

  if (enabled) {
    /* Increase progress */
    progressPercentage += units * unitPercentage;

    if (progressPercentage > 100.) {
      progressPercentage = 100.;
    }

    if (progress) {
      /*Notify the calling app that the progress changed*/
      if (progress->notify(msg, progressPercentage)) {
        return {Origin::CalibrationAlgorithm, ErrType::OperationAbortedByUser, "Calibration cancelled"};
      }
    }
  }

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
