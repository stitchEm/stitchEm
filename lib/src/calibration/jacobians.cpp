// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __clang_analyzer__  // VSA-7040

#include "jacobians.hpp"

namespace VideoStitch {
namespace Calibration {

void getJacobianRotationWrtYawPitchRoll(Eigen::Matrix<double, 9, 3>& J, double yaw, double pitch, double roll) {
  double x = pitch;
  double y = roll;
  double z = yaw;

  J.fill(0);

  J(0, 0) = -cos(z) * sin(y) + cos(y) * sin(x) * sin(z);
  J(0, 1) = cos(x) * sin(y) * sin(z);
  J(0, 2) = -cos(y) * sin(z) + cos(z) * sin(x) * sin(y);

  J(1, 0) = sin(y) * sin(z) + cos(y) * cos(z) * sin(x);
  J(1, 1) = cos(x) * cos(z) * sin(y);
  J(1, 2) = -cos(y) * cos(z) - sin(x) * sin(y) * sin(z);

  J(2, 0) = cos(x) * cos(y);
  J(2, 1) = -sin(x) * sin(y);

  J(3, 1) = -sin(x) * sin(z);
  J(3, 2) = cos(x) * cos(z);

  J(4, 1) = -cos(z) * sin(x);
  J(4, 2) = -cos(x) * sin(z);

  J(5, 1) = -cos(x);

  J(6, 0) = -cos(y) * cos(z) - sin(x) * sin(y) * sin(z);
  J(6, 1) = cos(x) * cos(y) * sin(z);
  J(6, 2) = sin(y) * sin(z) + cos(y) * cos(z) * sin(x);

  J(7, 0) = cos(y) * sin(z) - cos(z) * sin(x) * sin(y);
  J(7, 1) = cos(x) * cos(y) * cos(z);
  J(7, 2) = cos(z) * sin(y) - cos(y) * sin(x) * sin(z);

  J(8, 0) = -cos(x) * sin(y);
  J(8, 1) = -cos(y) * sin(x);
}

void getJacobianYawPitchRollWrtRotation(Eigen::Matrix<double, 3, 9>& J, const Eigen::Matrix3d& r) {
  J.fill(0);

  J(0, 2) = r(2, 2) / (r(2, 0) * r(2, 0) + r(2, 2) * r(2, 2));
  J(0, 8) = -r(2, 0) / (r(2, 0) * r(2, 0) + r(2, 2) * r(2, 2));
  J(1, 5) = -1.0 / sqrt(1.0 - r(2, 1) * r(2, 1));
  J(2, 3) = r(1, 1) / (r(0, 1) * r(0, 1) + r(1, 1) * r(1, 1));
  J(2, 4) = -r(0, 1) / (r(0, 1) * r(0, 1) + r(1, 1) * r(1, 1));
}

void getJacobianAxisAngleWrtRotation(Eigen::Matrix<double, 3, 9>& J, const Eigen::Matrix3d& R) {
  Eigen::Matrix3d eye;
  eye.setIdentity();

  Eigen::AngleAxisd aa(R);

  /*From rodrigues.m (by bouguet)*/
  double tr = 0.5 * (R.trace() - 1.0);
  double theta = acos(tr);

  if (aa.angle() < 1e-6) {
    J.fill(0);
    J(0, 5) = 0.5;
    J(0, 7) = -0.5;
    J(1, 2) = -0.5;
    J(1, 6) = 0.5;
    J(2, 1) = 0.5;
    J(2, 3) = -0.5;
    return;
  }

  Eigen::Matrix<double, 1, 9> dtrdR;
  dtrdR << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5;

  /*No need for huge precision here*/
  if (sin(theta) < 1e-8) {
    theta = 0.0001;
  }

  double dthetadtr = -1.0 / sqrt(1.0 - tr * tr);

  Eigen::Matrix<double, 1, 9> dthetadR = dthetadtr * dtrdR;

  double vth = 1.0 / (2.0 * sin(theta));

  double dvthdtheta = -vth * cos(theta) / sin(theta);
  Eigen::Matrix<double, 2, 1> dvar1dtheta;
  dvar1dtheta << dvthdtheta, 1.0;

  // var1 = [vth;theta];
  Eigen::Matrix<double, 2, 9> dvar1dR = (Eigen::Matrix<double, 2, 9>)(dvar1dtheta * dthetadR);

  Eigen::Matrix<double, 3, 1> om1;
  om1 << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);

  Eigen::Matrix<double, 3, 9> dom1dR;

  dom1dR << 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0;

  // var = [om1;vth;theta];

  Eigen::Matrix<double, 5, 9> dvardR;
  dvardR << dom1dR, dvar1dR;

  Eigen::Matrix<double, 3, 1> om = vth * om1;

  Eigen::Matrix<double, 3, 5> domdvar;
  Eigen::Vector3d zero31;
  zero31.fill(0);

  domdvar << vth * eye, om1, zero31;

  Eigen::Matrix<double, 1, 5> dthetadvar;
  dthetadvar << 0, 0, 0, 0, 1;

  Eigen::Matrix<double, 4, 5> dvar2dvar;
  dvar2dvar << domdvar, dthetadvar;

  Eigen::Matrix<double, 3, 4> domegadvar2;
  domegadvar2 << theta * eye, om;

  J = domegadvar2 * dvar2dvar * dvardR;
}

/** Jacobian tools */
void computedABtdA(Eigen::Matrix<double, 9, 9>& J, const Eigen::Matrix3d& /*A*/, const Eigen::Matrix3d& B) {
  J.fill(0.0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      J(i * 3 + 0, j * 3 + 0) = B(i, j);
      J(i * 3 + 1, j * 3 + 1) = B(i, j);
      J(i * 3 + 2, j * 3 + 2) = B(i, j);
    }
  }
}

/** Jacobian tools */
void computedABtdB(Eigen::Matrix<double, 9, 9>& J, const Eigen::Matrix3d& A, const Eigen::Matrix3d& /*B*/) {
  J.fill(0.0);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      J(i + 0, j * 3 + 0) = A(i, j);
      J(i + 3, j * 3 + 1) = A(i, j);
      J(i + 6, j * 3 + 2) = A(i, j);
    }
  }
}

}  // namespace Calibration
}  // namespace VideoStitch

#endif  // __clang_analyzer__
