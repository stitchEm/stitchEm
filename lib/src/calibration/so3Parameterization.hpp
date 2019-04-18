// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __SO3_PARAMETERZATION_HPP__
#define __SO3_PARAMETERZATION_HPP__

#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

class SO3Parameterization : public ceres::LocalParameterization {
 public:
  virtual ~SO3Parameterization() {}

  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    double* ptrBase = (double*)x;
    double* ptrResult = (double*)x_plus_delta;
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rotation(ptrBase);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rotationResult(ptrResult);

    Eigen::Vector3d axis;
    axis(0) = delta[0];
    axis(1) = delta[1];
    axis(2) = delta[2];
    double angle = axis.norm();
    axis.normalize();

    Eigen::AngleAxisd aa(angle, axis);
    Eigen::Matrix3d Rupdate;
    Rupdate = aa.toRotationMatrix();

    rotationResult = Rupdate * rotation;

    return true;
  }

  virtual bool ComputeJacobian(const double* /*x*/, double* jacobian) const {
    double* row[9];
    for (int i = 0; i < 9; i++) {
      row[i] = &jacobian[i * 3];
      for (int j = 0; j < 3; j++) {
        row[i][j] = 0;
      }
    }

    row[1][2] = 1;
    row[2][1] = -1;
    row[3][2] = -1;
    row[5][0] = 1;
    row[6][1] = 1;
    row[7][0] = -1;

    return true;
  }

  virtual int GlobalSize() const { return 9; }

  virtual int LocalSize() const { return 3; }
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
