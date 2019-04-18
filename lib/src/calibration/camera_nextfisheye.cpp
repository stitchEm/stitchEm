// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "camera_nextfisheye.hpp"

#include <iostream>

namespace VideoStitch {
namespace Calibration {

Camera* Camera_NextFisheye::clone() const {
  Camera_NextFisheye* ret = new Camera_NextFisheye;
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

/*Backproject to the unit sphere*/
bool Camera_NextFisheye::backproject(Eigen::Vector3d& campt, Eigen::Matrix<double, 3, 2>& jacobian,
                                     const Eigen::Vector2d& impt) {
  double x = impt(0);
  double y = impt(1);

  double len = x * x + y * y + 1.0;

  campt(0) = (2.0 * x) / len;
  campt(1) = (2.0 * y) / len;
  campt(2) = -(x * x + y * y - 1.0) / len;

  double s1 = 1.0 / (len * len);
  jacobian(0, 0) = 2.0 * (-x * x + y * y + 1.0) * s1;
  jacobian(0, 1) = -4.0 * x * y * s1;
  jacobian(1, 0) = -4.0 * x * y * s1;
  jacobian(1, 1) = 2.0 * (x * x - y * y + 1.0) * s1;
  jacobian(2, 0) = -4.0 * x * s1;
  jacobian(2, 1) = -4.0 * y * s1;

  return true;
}

/*Project from camera referencial*/
bool Camera_NextFisheye::project(Eigen::Vector2d& impt_meters, Eigen::Matrix<double, 2, 3>& jacobian,
                                 const Eigen::Vector3d& campt) {
  /*Point does not need to be on the unit sphere*/
  double len = campt.norm();
  double div = len + campt(2);

  if (fabs(div) < 1e-12) {
    return false;
  }

  impt_meters(0) = campt(0) / div;
  impt_meters(1) = campt(1) / div;

  jacobian.fill(0);
  jacobian(0, 0) = 1.0 / div;
  jacobian(1, 1) = 1.0 / div;
  jacobian(0, 2) = -campt(0) / (len * div);
  jacobian(1, 2) = -campt(1) / (len * div);

  return true;
}

}  // namespace Calibration
}  // namespace VideoStitch
