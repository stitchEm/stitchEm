// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef RANSAC_ROTATION_SOLVER_HPP_
#define RANSAC_ROTATION_SOLVER_HPP_

#include "rotationEstimation.hpp"
#include "eigengeometry.hpp"

#include "util/ransac.hpp"

#include <algorithm>
#include <iostream>
#include <random>

#define ENABLE_ROTATION_DISTANCE_AXIS_ANGLE 0
#define RANSAC_VERBOSE 0

namespace VideoStitch {
namespace Calibration {

class RansacRotationSolver : public Util::RansacSolver<RotationEstimationSolver> {
 public:
  RansacRotationSolver(const RotationEstimationProblem& problem, const Eigen::Matrix3d& mean,
                       const Eigen::Matrix3d& cov, int minSamplesForFit, int numIters, int minConsensusSamples,
                       double angleThreshold, std::default_random_engine* gen = nullptr, bool debug = false,
                       bool useFloatPrecision = false)
      : Util::RansacSolver<RotationEstimationSolver>(problem, minSamplesForFit, numIters, minConsensusSamples, gen,
                                                     debug, useFloatPrecision),
        angleThreshold(angleThreshold),
        mean(mean) {
    invcov = cov.inverse();
  }

 private:
  virtual bool isConsensualSample(double* values) const { return (*values < angleThreshold); }

  virtual bool validate(double* params) const {
#if ENABLE_ROTATION_DISTANCE_AXIS_ANGLE
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R(i, j) = params[i * 3 + j];
      }
    }

    Eigen::Matrix3d Rd = R * mean.transpose();
    Eigen::AngleAxisd aa(Rd);

    Eigen::Vector3d diff = aa.axis() * aa.angle();

#ifdef RANSAC_VERBOSE
    std::cout << "mahanalobis from presets for AngleAxis " << diff.transpose() * invcov * diff << std::endl;
#endif

    double mahalanobis = diff.transpose() * invcov * diff;

    // using the 3.sigmas rule for Normal distributions: the Manhanalobis distance is a squared norm, use 3.0^2 as the
    // threshold
    if (mahalanobis > 9.0) {
      return false;
    }
#else
    // suppress unused parameter warning
    (void)params;
#endif

    return true;
  }

  const double angleThreshold;
  Eigen::Matrix3d mean;
  Eigen::Matrix3d invcov;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
