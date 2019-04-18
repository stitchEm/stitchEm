// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "camera_perspective.hpp"

#include <iostream>

namespace VideoStitch {
namespace Calibration {

Camera* Camera_Perspective::clone() const {
  Camera_Perspective* ret = new Camera_Perspective;
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

bool Camera_Perspective::backproject(Eigen::Vector3d& campt, Eigen::Matrix<double, 3, 2>& jacobian,
                                     const Eigen::Vector2d& impt) {
  double x = impt(0);
  double y = impt(1);

  double len = sqrt(x * x + y * y + 1.0);
  campt(0) = x / len;
  campt(1) = y / len;
  campt(2) = 1.0 / len;

  double s1 = 1.0 / (len * len * len);
  jacobian(0, 0) = (y * y + 1.0) * s1;
  jacobian(0, 1) = -x * y * s1;
  jacobian(1, 0) = -x * y * s1;
  jacobian(1, 1) = (x * x + 1.0) * s1;
  jacobian(2, 0) = -x * s1;
  jacobian(2, 1) = -y * s1;

  return true;
}

bool Camera_Perspective::project(Eigen::Vector2d& impt_meters, Eigen::Matrix<double, 2, 3>& jacobian,
                                 const Eigen::Vector3d& campt) {
  if (campt(2) > .0001) {
    impt_meters(0) = campt(0) / campt(2);
    impt_meters(1) = campt(1) / campt(2);

    jacobian.fill(0);
    jacobian(0, 0) = 1.0 / campt(2);
    jacobian(1, 1) = 1.0 / campt(2);
    jacobian(0, 2) = -campt(0) / (campt(2) * campt(2));
    jacobian(1, 2) = -campt(1) / (campt(2) * campt(2));
  } else {
    impt_meters(0) = -100.;
    impt_meters(1) = -100.;

    jacobian.fill(0);
  }

  return true;
}

}  // namespace Calibration
}  // namespace VideoStitch
