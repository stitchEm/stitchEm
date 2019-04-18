// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CALIBRATION_PROGRESS_HPP_
#define CALIBRATION_PROGRESS_HPP_

#include "libvideostitch/algorithm.hpp"

#include <mutex>

namespace VideoStitch {
namespace Calibration {

/**
 * Simple helper class to ease the progress bar updating progress
 * for the automatic calibration algorithm
 */
class VS_EXPORT CalibrationProgress {
 public:
  /**
  @brief Constructor
  @param progress Algorithm progress callback object
  @param totalUnits number of units to report 100% progress
  */
  CalibrationProgress(VideoStitch::Util::Algorithm::ProgressReporter* progress, const double totalUnits);

  /**
  @brief Copy constructor
  @param other the source of copy
  */
  CalibrationProgress(const CalibrationProgress& other);

  /**
  @brief Add a progress step
  @param units number of progress done (respective to total units)
  @param msg the description sent to the reporter
  */
  Status add(double units, const std::string& msg = "Calibration");

  /**
  @brief Enables the progress reporting (enabled by default)
  */
  void enable();

  /**
  @brief Disables the progress reporting (enabled by default)
  */
  void disable();

 private:
  VideoStitch::Util::Algorithm::ProgressReporter* progress;
  double progressPercentage;
  double unitPercentage;
  std::mutex progressLock;
  bool enabled;

 public:
  const static double seek;
  const static double kpDetect;
  const static double kpMatch;
  const static double fovIterate;
  const static double deshuffle;
  const static double filter;
  const static double initGeometry;
  const static double optim;
  const static double optim_done;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
