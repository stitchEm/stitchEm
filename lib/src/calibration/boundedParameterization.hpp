// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __BOUNDED_PARAMETERZATION_HPP__
#define __BOUNDED_PARAMETERZATION_HPP__

#include <ceres/ceres.h>

namespace VideoStitch {
namespace Calibration {

class BoundedParameterization : public ceres::LocalParameterization {
 public:
  virtual ~BoundedParameterization() {}

  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    double* ptrBase = (double*)x;
    double* ptrResult = (double*)x_plus_delta;
    Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > base(ptrBase);
    Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > result(ptrResult);

    double update = delta[0];

    Eigen::Matrix<double, 2, 2> Mupdate;
    Mupdate(0, 0) = cos(update);
    Mupdate(0, 1) = -sin(update);
    Mupdate(1, 0) = sin(update);
    Mupdate(1, 1) = cos(update);

    result = Mupdate * base;

    return true;
  }

  /*Constant jacobian*/
  virtual bool ComputeJacobian(const double* /*x*/, double* jacobian) const {
    jacobian[0] = 0;
    jacobian[1] = 1;
    jacobian[2] = -1;
    jacobian[3] = 0;

    return true;
  }

  virtual int GlobalSize() const { return 4; }

  virtual int LocalSize() const { return 1; }
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
