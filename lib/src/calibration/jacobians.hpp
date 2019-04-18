// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __JACOBIANS__HPP
#define __JACOBIANS__HPP

#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

/**
@brief Compute the jacobian of the function which computes a rotation matrix from yaw pitch and roll wrt yaw pitch roll
@param J the output jacobian matrix
@param yaw the yaw rotation input parameter
@param pitch the pitch rotation input parameter
@param roll the roll rotation input parameter
*/
void getJacobianRotationWrtYawPitchRoll(Eigen::Matrix<double, 9, 3>& J, double yaw, double pitch, double roll);

/**
@brief Compute the jacobian of the function which computes yaw pitch roll from a rotation matrix wrt the rotation matrix
@param J the output jacobian
@param r the input rotation matrix
*/
void getJacobianYawPitchRollWrtRotation(Eigen::Matrix<double, 3, 9>& J, const Eigen::Matrix3d& r);

/**
@brief Compute the jacobian of the logSO3 wrt the input rotation matrix
@param J the output jacobian
@param R the input rotation matrix
*/
void getJacobianAxisAngleWrtRotation(Eigen::Matrix<double, 3, 9>& J, const Eigen::Matrix3d& R);

/**
@brief Compute the jacobian of matrix multiplication A*B.t() wrt A
@param J the result jacobian
@param A the first operand
@param B the second operand
*/
void computedABtdA(Eigen::Matrix<double, 9, 9>& J, const Eigen::Matrix3d& A, const Eigen::Matrix3d& B);

/**
@brief Compute the jacobian of matrix multiplication A*B.t() wrt B
@param J the result jacobian
@param A the first operand
@param B the second operand
*/
void computedABtdB(Eigen::Matrix<double, 9, 9>& J, const Eigen::Matrix3d& A, const Eigen::Matrix3d& B);

}  // namespace Calibration
}  // namespace VideoStitch

#endif
