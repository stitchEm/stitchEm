// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "controlPointFilter.hpp"

#include "camera.hpp"
#include "rotationEstimation.hpp"
#include "ransacRotationSolver.hpp"
#include "calibrationUtils.hpp"

#include "common/angles.hpp"
#include "core/geoTransform.hpp"

#include "libvideostitch/logging.hpp"

#include <memory>

namespace VideoStitch {
namespace Calibration {

ControlPointFilter::ControlPointFilter(double cellFactor, double angleThreshold, double minRatioInliers,
                                       int minSamplesForFit, double ratioOutliers, double probaDrawOutlierFreeSample)
    : score(0.0),
      consensus(0),
      cellFactor(cellFactor),
      angleThreshold(angleThreshold),
      minRatioInliers(minRatioInliers),
      minSamplesForFit(minSamplesForFit) {
  numIters =
      (int)ceil(log(1.0 - probaDrawOutlierFreeSample) / log(1.0 - pow(1.0 - ratioOutliers, (double)minSamplesForFit)));
}

bool ControlPointFilter::filterFromExtrinsics(Core::ControlPointList& filteredControlPoints,
                                              const std::shared_ptr<Camera>& camera1,
                                              const std::shared_ptr<Camera>& camera2,
                                              const Core::ControlPointList& currentControlPoints,
                                              const Core::ControlPointList& formerControlPoints,
                                              const Core::ControlPointList& syntheticControlPoints,
                                              std::default_random_engine& gen) {
  Core::ControlPointList sortedControlPoints(currentControlPoints);
  Core::ControlPointList sortedFormerControlPoints(formerControlPoints);
  Core::ControlPointList decimatedControlPoints;

  filteredControlPoints.clear();

  sortedControlPoints.sort(Core::ControlPointComparator());
  sortedFormerControlPoints.sort(Core::ControlPointComparator());
  // no need to sort synthetic control points, they all have the same score

  // current control points take precedence over former control points and synthetic ones
  // append the former ones to the current ones and let them go through the decimation process
  sortedControlPoints.insert(sortedControlPoints.end(), sortedFormerControlPoints.begin(),
                             sortedFormerControlPoints.end());
  sortedControlPoints.insert(sortedControlPoints.end(), syntheticControlPoints.begin(), syntheticControlPoints.end());

  // initialize score and consensus values
  score = std::numeric_limits<double>::max();
  consensus = 0;

  decimateSortedControlPoints(decimatedControlPoints, sortedControlPoints, camera1->getWidth(), camera1->getHeight(),
                              cellFactor);

  const size_t numberCP = decimatedControlPoints.size();
  if (numberCP < (size_t)minSamplesForFit) {
    Logger::get(Logger::Verbose) << "Not enough control points for Ransac" << std::endl;
    return false;
  }

  MatchList matchList;

  int indexDecimatedPoint = 0;
  for (auto& c : decimatedControlPoints) {
    Eigen::Vector3d v1, v2;
    Eigen::Vector2d pt;
    pt(0) = c.x0;
    pt(1) = c.y0;
    camera1->quicklift(v1, pt);

    pt(0) = c.x1;
    pt(1) = c.y1;
    camera2->quicklift(v2, pt);

    Logger::get(Logger::Debug) << "  DecimatedPoint(" << indexDecimatedPoint++ << "): ";
    Logger::get(Logger::Debug) << "   v1: " << v1(0) << " " << v1(1) << " " << v1(2) << "    v2: " << v2(0) << " "
                               << v2(1) << " " << v2(2) << std::endl;

    matchList.push_back(SpherePointMatch(v1, v2));
  }

  Eigen::Matrix3d second_Rmean_first;
  Eigen::Matrix3d second_angleAxisRcov_first;
  Camera::getRelativeRotation(second_Rmean_first, second_angleAxisRcov_first, *camera1, *camera2);

  Logger::get(Logger::Debug) << "second_Rmean_first: " << second_Rmean_first(0, 0) << " " << second_Rmean_first(0, 1)
                             << " " << second_Rmean_first(0, 2) << "   " << second_Rmean_first(1, 0) << " "
                             << second_Rmean_first(1, 1) << " " << second_Rmean_first(1, 2) << "   "
                             << second_Rmean_first(2, 0) << " " << second_Rmean_first(2, 1) << " "
                             << second_Rmean_first(2, 2) << std::endl;
  Logger::get(Logger::Debug) << "second_Rcov_first: " << second_angleAxisRcov_first(0, 0) << " "
                             << second_angleAxisRcov_first(0, 1) << " " << second_angleAxisRcov_first(0, 2) << "   "
                             << second_angleAxisRcov_first(1, 0) << " " << second_angleAxisRcov_first(1, 1) << " "
                             << second_angleAxisRcov_first(1, 2) << "   " << second_angleAxisRcov_first(2, 0) << " "
                             << second_angleAxisRcov_first(2, 1) << " " << second_angleAxisRcov_first(2, 2)
                             << std::endl;

  RotationEstimationProblem problem(matchList);

  const int minConsensusSamples = (int)floor(minRatioInliers * (double)numberCP);
  const float inlierTolerance = (float)degToRad(angleThreshold);

  RansacRotationSolver ransacRotationSolver(problem, second_Rmean_first, second_angleAxisRcov_first, minSamplesForFit,
                                            numIters, minConsensusSamples, inlierTolerance, &gen, false, false);

  std::vector<double> params(9);
  std::vector<char> inlierIndices;
  std::vector<double> outputResiduals;
  if (!ransacRotationSolver.run(params, inlierIndices, outputResiduals)) {
    Logger::get(Logger::Verbose) << "Ransac filtering did not converge" << std::endl;
    return false;
  }

  score = 0.0;
  size_t idx = 0;
  for (auto& cp : decimatedControlPoints) {
    if (inlierIndices[idx]) {
      score += std::abs(outputResiduals[idx]);
      filteredControlPoints.push_back(cp);
    }
    idx++;
  }
  score /= (double)filteredControlPoints.size();
  consensus = idx;

  if (filteredControlPoints.empty()) {
    Logger::get(Logger::Verbose) << "No inlier control points found" << std::endl;
    return false;
  }

  int pos = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      estimatedR(i, j) = params[pos];
      pos++;
    }
  }

  return true;
}

Status ControlPointFilter::projectFromEstimatedRotation(Core::ControlPointList& filteredControlPoints,
                                                        const std::shared_ptr<Camera>& camera1,
                                                        const std::shared_ptr<Camera>& camera2) {
  for (auto& c : filteredControlPoints) {
    Eigen::Vector3d v1, v2;
    Eigen::Vector2d pt;
    bool res;

    // init reprojected points
    c.rx0 = c.ry0 = c.rx1 = c.ry1 = 0.;

    // quicklift point from camera 1
    pt(0) = c.x0;
    pt(1) = c.y0;
    res = camera1->quicklift(v1, pt);
    if (!res) {
      continue;
    }

    // go from camera1 to camera2
    v1 = estimatedR * v1;

    // project from camera space to camera plane
    res = camera2->quickproject(pt, v1);
    if (!res) {
      continue;
    }
    c.rx0 = pt(0);
    c.ry0 = pt(1);

    // quicklift point from camera2
    pt(0) = c.x1;
    pt(1) = c.y1;
    res = camera2->quicklift(v2, pt);
    if (!res) {
      continue;
    }

    // go from camera2 to camera1
    v2 = estimatedR.transpose() * v2;

    // project from camera space to camera plane
    res = camera1->quickproject(pt, v2);
    if (!res) {
      continue;
    }
    c.rx1 = pt(0);
    c.ry1 = pt(1);
  }

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
