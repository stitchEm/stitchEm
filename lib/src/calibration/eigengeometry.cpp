// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "eigengeometry.hpp"

#include <Eigen/Geometry>

namespace VideoStitch {
namespace Calibration {

#ifndef __clang_analyzer__  // VSA-7040
void rotationFromEulerZXY(Eigen::Matrix3d& R, double yaw, double pitch, double roll) {
  Eigen::AngleAxisd X(-pitch, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd Y(-yaw, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd Z(-roll, Eigen::Vector3d::UnitZ());

  R = Z.matrix() * X.matrix() * Y.matrix();
}
#endif  // __clang_analyzer__

void EulerZXYFromRotation(Eigen::Vector3d& vr, const Eigen::Matrix3d& R) {
  vr(0) = atan2(R(2, 0), R(2, 2));
  vr(1) = -asin(R(2, 1));
  vr(2) = atan2(R(0, 1), R(1, 1));
}

}  // namespace Calibration
}  // namespace VideoStitch
