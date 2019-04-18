// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mergerMaskProgress.hpp"

#include <iostream>

namespace VideoStitch {
namespace MergerMask {

MergerMaskProgress::MergerMaskProgress(VideoStitch::Util::Algorithm::ProgressReporter* progress,
                                       const size_t numCameras, const bool blendingOrder, const bool findSeam)
    : progress(progress), progressPercentage(0.0), unitPercentage(0.0) {
  progressComponents["coarseMask_setup"] = std::make_pair(1, 1.0);
  progressComponents["coarseMask_lowRes"] =
      blendingOrder ? std::make_pair(numCameras * (numCameras - 1), 0.5) : std::make_pair(numCameras, 0.5);
  progressComponents["coarseMask_fulRes"] = std::make_pair(1, 1.0);
  if (findSeam) {
    progressComponents["seam_setup"] = std::make_pair(1, 0.2);
    progressComponents["seam_fulRes"] = std::make_pair(numCameras, 10.0);
    progressComponents["seam_updateMask"] = std::make_pair(1, 1.0);
  }

  double sum = 0.0;
  for (auto component : progressComponents) {
    sum += component.second.first * component.second.second;
  }
  unitPercentage = 100.0 / sum;
}

MergerMaskProgress::MergerMaskProgress(const MergerMaskProgress& other)
    : progress(other.progress),
      progressPercentage(other.progressPercentage),
      unitPercentage(other.unitPercentage),
      progressComponents(other.progressComponents),
      processedComponents(other.processedComponents) {}

Status MergerMaskProgress::add(const std::string& name, const std::string& msg) {
  std::lock_guard<std::mutex> lock(progressLock);  // ensure thread-safety
  /* Look for the right unit */
  if (progressComponents.find(name) == progressComponents.end()) {
    return {Origin::Stitcher, ErrType::ImplementationError, "Internal merger mask error"};
  }
  if (processedComponents.find(name) == processedComponents.end()) {
    processedComponents[name] = 0;
  }
  const size_t processedUnit = processedComponents[name];
  if (processedUnit >= progressComponents[name].first) {
    return {Origin::Stitcher, ErrType::ImplementationError, "Internal merger mask processing error"};
  }
  const double units = progressComponents[name].second;
  processedComponents[name] = processedComponents[name] + 1;
  /* Increase progress */
  progressPercentage += units * unitPercentage;

  if (progress) {
    /*Notify the calling app that the progress changed*/
    if (progress->notify(msg, progressPercentage)) {
      return Status{Origin::BlendingMaskAlgorithm, ErrType::OperationAbortedByUser, "Algorithm cancelled"};
    }
  }
  return Status::OK();
}

}  // namespace MergerMask
}  // namespace VideoStitch
