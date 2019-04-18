// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "camera_fisheye.hpp"
#include <iostream>

namespace VideoStitch {
namespace Calibration {

Camera* Camera_Fisheye::clone() const {
  Camera_Fisheye* ret = new Camera_Fisheye;
  ret->yprCovariance = yprCovariance;
  ret->cameraRreference = cameraRreference;
  ret->cameraRreference_covariance = cameraRreference_covariance;
  ret->cameraR = cameraR;
  ret->horizontal_focal = horizontal_focal;
  ret->vertical_focal = vertical_focal;
  ret->horizontal_center = horizontal_center;
  ret->vertical_center = vertical_center;
  ret->distort_A = distort_A;
  ret->distort_B = distort_B;
  ret->distort_C = distort_C;
  ret->t_x = t_x;
  ret->t_y = t_y;
  ret->t_z = t_z;
  ret->format = format;
  ret->width = width;
  ret->height = height;

  return ret;
}

bool Camera_Fisheye::backproject(Eigen::Vector3d& campt, Eigen::Matrix<double, 3, 2>& jacobian,
                                 const Eigen::Vector2d& impt) {
  double x = impt(0);
  double y = impt(1);

  double phi = atan2(y, x);
  double theta = sqrt(x * x + y * y);

  campt(0) = sin(theta) * cos(phi);
  campt(1) = sin(theta) * sin(phi);
  campt(2) = cos(theta);

  jacobian.fill(0);

  if (theta < 1e-12) {
    return false;
  }

  double dphidx = -y / (x * x + y * y);
  double dphidy = x / (x * x + y * y);
  double dthetadx = x / sqrt(x * x + y * y);
  double dthetady = y / sqrt(x * x + y * y);

  double dcosphidphi = -sin(phi);
  double dsinphidphi = cos(phi);
  double dcosthetadtheta = -sin(theta);
  double dsinthetadtheta = cos(theta);

  double dcosphidx = dcosphidphi * dphidx;
  double dcosphidy = dcosphidphi * dphidy;
  double dcosthetadx = dcosthetadtheta * dthetadx;
  double dcosthetady = dcosthetadtheta * dthetady;
  double dsinphidx = dsinphidphi * dphidx;
  double dsinphidy = dsinphidphi * dphidy;
  double dsinthetadx = dsinthetadtheta * dthetadx;
  double dsinthetady = dsinthetadtheta * dthetady;

  jacobian(0, 0) = dsinthetadx * cos(phi) + sin(theta) * dcosphidx;
  jacobian(0, 1) = dsinthetady * cos(phi) + sin(theta) * dcosphidy;
  jacobian(1, 0) = dsinthetadx * sin(phi) + sin(theta) * dsinphidx;
  jacobian(1, 1) = dsinthetady * sin(phi) + sin(theta) * dsinphidy;
  jacobian(2, 0) = dcosthetadx;
  jacobian(2, 1) = dcosthetady;

  return true;
}

bool Camera_Fisheye::project(Eigen::Vector2d& impt_meters, Eigen::Matrix<double, 2, 3>& jacobian,
                             const Eigen::Vector3d& campt) {
  double X = campt(0);
  double Y = campt(1);
  double Z = campt(2);
  double len = std::sqrt(X * X + Y * Y + Z * Z);

  if (len <= 1e-6) {
    return false;
  }

  double theta = (len > 0.) ? acos(Z / len) : 0.;
  double phi = atan2(Y, X);
  impt_meters(0) = cos(phi) * theta;
  impt_meters(1) = sin(phi) * theta;

  jacobian.fill(0);

  double dthetadx = X * Z / (std::sqrt(X * X + Y * Y) * len * len);
  double dthetady = Y * Z / (std::sqrt(X * X + Y * Y) * len * len);
  double dthetadz = -std::sqrt(X * X + Y * Y) / (len * len);
  double dphidx = -Y / (X * X + Y * Y);
  double dphidy = X / (X * X + Y * Y);
  double dphidz = 0;

  double dcosdphi = -sin(phi);
  double dsindphi = cos(phi);

  double dcosdx = dcosdphi * dphidx;
  double dcosdy = dcosdphi * dphidy;
  double dcosdz = dcosdphi * dphidz;
  double dsindx = dsindphi * dphidx;
  double dsindy = dsindphi * dphidy;
  double dsindz = dsindphi * dphidz;

  jacobian(0, 0) = cos(phi) * dthetadx + dcosdx * theta;
  jacobian(0, 1) = cos(phi) * dthetady + dcosdy * theta;
  jacobian(0, 2) = cos(phi) * dthetadz + dcosdz * theta;
  jacobian(1, 0) = sin(phi) * dthetadx + dsindx * theta;
  jacobian(1, 1) = sin(phi) * dthetady + dsindy * theta;
  jacobian(1, 2) = sin(phi) * dthetadz + dsindz * theta;

  return true;
}

}  // namespace Calibration
}  // namespace VideoStitch
