// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"

#include <Eigen/Dense>

#include <random>

namespace VideoStitch {
namespace Calibration {

class Camera;

/**
@brief  This class is used to filter outlier control points.
@details Given two inputs and the control points between them, it finds the best model to align two images in the
panorama
@details Returns the optimal relative position of the images and a list of control points containing only the inlier
control points
*/
class ControlPointFilter {
 public:
  ControlPointFilter(double cellFactor, double angleThreshold, double minRatioInliers, int minSamplesForFit,
                     double ratioOutliers, double probaDrawOutlierFreeSample);

  /**
   * @brief Uses the control points to align inputs.
   * @param filteredControlPoints the returned list of filtered control points
   * @param camera1 the first camera object
   * @param camera2 the second camera object
   * @param currentControlPoints the list of control points extracted from the current input pictures
   * @param formerControlPoints the list of control points coming from the PanoDefinition (extracted during a former
   * calibration)
   * @param syntheticControlPoints the list of synthetic control points generated to cover the input areas where no
   * control point has been extracted
   * @param gen the random number generator
   * @note currentControlPoints take precedence over formerControlPoints, which take precedence over
   * syntheticControlPoints
   * @return true on success
   * @return false on failure
   */
  bool filterFromExtrinsics(Core::ControlPointList& filteredControlPoints, const std::shared_ptr<Camera>& camera1,
                            const std::shared_ptr<Camera>& camera2, const Core::ControlPointList& currentControlPoints,
                            const Core::ControlPointList& formerControlPoints,
                            const Core::ControlPointList& syntheticControlPoints, std::default_random_engine& gen);

  /**
   * @brief Uses the estimated rotation to project the control points.
   * @return true on success
   * @return false on failure
   */
  Status projectFromEstimatedRotation(Core::ControlPointList& filteredControlPoints,
                                      const std::shared_ptr<Camera>& camera1, const std::shared_ptr<Camera>& camera2);

  Eigen::Matrix3d getEstimatedRotation() { return estimatedR; }

  /**
   * @brief Get rotation score
   * @return sum of angular distances between rotated inliers
   */
  double getScore() { return score; }

  /**
   * @brief gets the number of inliers with found rotation
   * @return number of inliers with found rotation
   */
  size_t getConsensus() { return consensus; }

 private:
  Eigen::Matrix3d estimatedR;
  double score;
  size_t consensus;

 private:
  double cellFactor;       // Cell for homogeneous distribution
  double angleThreshold;   // max angular distance between two reprojected CPs to be considered as inliers, in degrees
  double minRatioInliers;  // minimum ratio of inliers that need to be found to validate a model
  int minSamplesForFit;    // minimum number of control points needed to estimate a model
  int numIters;            // number of RANSAC iterations
};

}  // namespace Calibration
}  // namespace VideoStitch
