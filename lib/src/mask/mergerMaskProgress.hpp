// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"

#include <mutex>
#include <unordered_map>

namespace VideoStitch {
namespace MergerMask {

/**
 * Simple helper class to ease the progress bar updating progress
 * for the automatic blending mask algorithm
 */
class MergerMaskProgress {
 public:
  /**
   * Constructor
   * @param progress Algorithm progress callback object
   * @param numCameras number of cameras on the rig
   * @param numFrames number of images seen by one camera
   */
  MergerMaskProgress(VideoStitch::Util::Algorithm::ProgressReporter* progress, const size_t numCameras,
                     const bool blendingOrder, const bool findSeam);

  /**
   * Copy constructor
   * @param other the source of copy
   */
  MergerMaskProgress(const MergerMaskProgress& other);

  /**
   * Add a progress step
   * @param units name of the processed unit
   * @param msg the description sent to the reporter
   */
  Status add(const std::string& name, const std::string& msg = "Blending mask");

 private:
  VideoStitch::Util::Algorithm::ProgressReporter* progress;
  double progressPercentage;
  double unitPercentage;
  std::unordered_map<std::string, std::pair<size_t, double>> progressComponents;
  std::unordered_map<std::string, size_t> processedComponents;

  std::mutex progressLock;
};

}  // namespace MergerMask
}  // namespace VideoStitch
