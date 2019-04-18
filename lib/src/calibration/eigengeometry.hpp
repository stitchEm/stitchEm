// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __EIGEN_GEOMETRY__
#define __EIGEN_GEOMETRY__

#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

/**
@brief Create a rotation matrix from a yaw pitch roll set of parameters
@param R the output rotation matrix
@param yaw the input yaw parameter
@param pitch the input yaw parameter
@param roll the input yaw parameter
*/
void rotationFromEulerZXY(Eigen::Matrix3d& R, double yaw, double pitch, double roll);

/**
@brief Compute yaw pitch roll vector from a rotation matrix
@param vr the output 3d vector (yaw; pitch; roll)
@param R the input rotation matrix
*/
void EulerZXYFromRotation(Eigen::Vector3d& vr, const Eigen::Matrix3d& R);

}  // namespace Calibration
}  // namespace VideoStitch
#endif
