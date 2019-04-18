// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "camera.hpp"

#include "libvideostitch/logging.hpp"

#include <ceres/ceres.h>

namespace VideoStitch {
namespace Calibration {

class rotationDistanceCostFunction : public ceres::CostFunction {
 public:
  explicit rotationDistanceCostFunction(const std::shared_ptr<const Camera>& cam) : cam(cam) {
    set_num_residuals(1);
    std::vector<int>* blocks = mutable_parameter_block_sizes();
    blocks->push_back(9);
  }

  /*
   * Evaluate() implements a "hard" limit on rotations, not a smooth distance function:
   * if the current rotation is out of allowed presets, returns false to inform ceres not to go for this rotation
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> cameraR(parameters[0]);

    residuals[0] = 0;

    if (!cam->isRotationWithinPresets(cameraR)) {
      Logger::get(Logger::Warning) << "out of bound rotation, returning false" << std::endl;
      // return false to tell ceres that this rotation is not valid
      return false;
    }

    residuals[0] = 0;

    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 1, 9, Eigen::RowMajor>> J(jacobians[0]);

        J.setZero();
      }
    }

    return true;
  }

 private:
  const std::shared_ptr<const Camera> cam;
};

}  // namespace Calibration
}  // namespace VideoStitch
