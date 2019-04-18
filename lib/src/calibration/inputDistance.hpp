// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __INPUT_DISTANCE_HPP__
#define __INPUT_DISTANCE_HPP__

#include "calibrationConfig.hpp"
#include "camera.hpp"

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <unordered_map>
#include <assert.h>

namespace VideoStitch {
namespace Calibration {

#define INDEX_HFOCAL_CAM1 0
#define INDEX_HFOCAL_CAM2 1
#define INDEX_VFOCAL_CAM1 2
#define INDEX_VFOCAL_CAM2 3
#define INDEX_HCENTER_CAM1 4
#define INDEX_HCENTER_CAM2 5
#define INDEX_VCENTER_CAM1 6
#define INDEX_VCENTER_CAM2 7
#define INDEX_LENSDISTORTA_CAM1 8
#define INDEX_LENSDISTORTA_CAM2 9
#define INDEX_LENSDISTORTB_CAM1 10
#define INDEX_LENSDISTORTB_CAM2 11
#define INDEX_LENSDISTORTC_CAM1 12
#define INDEX_LENSDISTORTC_CAM2 13
#define INDEX_ROTATION_CAM1 14
#define INDEX_ROTATION_CAM2 15
#define INDEX_TX_CAM1 16
#define INDEX_TX_CAM2 17
#define INDEX_TY_CAM1 18
#define INDEX_TY_CAM2 19
#define INDEX_TZ_CAM1 20
#define INDEX_TZ_CAM2 21
#define INDEX_3DPOINT 22
#define INDEX_LAST 23

#define SIZE_VFOCAL_CAM1 4
#define SIZE_VFOCAL_CAM2 4
#define SIZE_HFOCAL_CAM1 4
#define SIZE_HFOCAL_CAM2 4
#define SIZE_VCENTER_CAM1 4
#define SIZE_VCENTER_CAM2 4
#define SIZE_HCENTER_CAM1 4
#define SIZE_HCENTER_CAM2 4
#define SIZE_LENSDISTORTA_CAM1 4
#define SIZE_LENSDISTORTA_CAM2 4
#define SIZE_LENSDISTORTB_CAM1 4
#define SIZE_LENSDISTORTB_CAM2 4
#define SIZE_LENSDISTORTC_CAM1 4
#define SIZE_LENSDISTORTC_CAM2 4
#define SIZE_ROTATION_CAM1 9
#define SIZE_ROTATION_CAM2 9
#define SIZE_TX_CAM1 4
#define SIZE_TX_CAM2 4
#define SIZE_TY_CAM1 4
#define SIZE_TY_CAM2 4
#define SIZE_TZ_CAM1 4
#define SIZE_TZ_CAM2 4
#define SIZE_3DPOINT 3

class inputDistanceCostFunction : public ceres::CostFunction {
 public:
  inputDistanceCostFunction(const std::shared_ptr<Camera>& cam1, const std::shared_ptr<Camera>& cam2,
                            const Eigen::Vector2d& impt1, const Eigen::Vector2d& impt2, const CalibrationConfig& config,
                            const double sphereScale)
      : camera1(cam1->clone()),
        camera2(cam2->clone()),
        impt1(impt1),
        impt2(impt2),
        sphereScale(sphereScale),
        needsScaling(std::abs(sphereScale - 1.) > 1e-6) {
    // prepare hashmap of parameter indexes, to get continuously increasing indices and parameter availability through
    // the has() function not all of them may be used in a cost function, depending on the calibration config this
    // hashmap is used to find
    hashmap.reserve(INDEX_LAST);
    int index = 0;

    hashmap[INDEX_HFOCAL_CAM1] = index++;
    if (!config.hasSingleFocal()) {
      hashmap[INDEX_HFOCAL_CAM2] = index++;
    }
    hashmap[INDEX_VFOCAL_CAM1] = index++;
    if (!config.hasSingleFocal()) {
      hashmap[INDEX_VFOCAL_CAM2] = index++;
    }
    hashmap[INDEX_HCENTER_CAM1] = index++;
    hashmap[INDEX_HCENTER_CAM2] = index++;
    hashmap[INDEX_VCENTER_CAM1] = index++;
    hashmap[INDEX_VCENTER_CAM2] = index++;
    hashmap[INDEX_LENSDISTORTA_CAM1] = index++;
    hashmap[INDEX_LENSDISTORTA_CAM2] = index++;
    hashmap[INDEX_LENSDISTORTB_CAM1] = index++;
    hashmap[INDEX_LENSDISTORTB_CAM2] = index++;
    hashmap[INDEX_LENSDISTORTC_CAM1] = index++;
    hashmap[INDEX_LENSDISTORTC_CAM2] = index++;
    hashmap[INDEX_ROTATION_CAM1] = index++;
    hashmap[INDEX_ROTATION_CAM2] = index++;
    assert(index <= INDEX_LAST);

    set_num_residuals(2);
    std::vector<int>* blocks = mutable_parameter_block_sizes();

    // adds the block params if index is in hashmap
#define CONDITIONAL_ADD_BLOCK(PARAM) \
  if (has(INDEX_##PARAM)) blocks->push_back(SIZE_##PARAM)

    CONDITIONAL_ADD_BLOCK(HFOCAL_CAM1);
    CONDITIONAL_ADD_BLOCK(HFOCAL_CAM2);
    CONDITIONAL_ADD_BLOCK(VFOCAL_CAM1);
    CONDITIONAL_ADD_BLOCK(VFOCAL_CAM2);
    CONDITIONAL_ADD_BLOCK(HCENTER_CAM1);
    CONDITIONAL_ADD_BLOCK(HCENTER_CAM2);
    CONDITIONAL_ADD_BLOCK(VCENTER_CAM1);
    CONDITIONAL_ADD_BLOCK(VCENTER_CAM2);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTA_CAM1);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTA_CAM2);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTB_CAM1);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTB_CAM2);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTC_CAM1);
    CONDITIONAL_ADD_BLOCK(LENSDISTORTC_CAM2);
    CONDITIONAL_ADD_BLOCK(ROTATION_CAM1);
    CONDITIONAL_ADD_BLOCK(ROTATION_CAM2);

#undef CONDITIONAL_ADD_BLOCK
  }

  bool has(char parameter_index) const {
    assert(parameter_index < INDEX_LAST);
    return hashmap.find(parameter_index) != hashmap.end();
  }

  size_t map(char parameter_index) const {
    assert(parameter_index < INDEX_LAST);
    return hashmap.at(parameter_index);
  }

  void resetJacobians(double** jacobians) const {
    if (jacobians == nullptr) {
      return;
    }

#define RESET_JACOBIAN(PARAM, ROWS, COLS)                             \
  if (has(INDEX_##PARAM) && jacobians[map(INDEX_##PARAM)] != nullptr) \
  Eigen::Map<Eigen::Matrix<double, ROWS, COLS, Eigen::RowMajor> >(jacobians[map(INDEX_##PARAM)]).fill(0)

    RESET_JACOBIAN(HFOCAL_CAM1, 2, 4);
    RESET_JACOBIAN(HFOCAL_CAM2, 2, 4);
    RESET_JACOBIAN(VFOCAL_CAM1, 2, 4);
    RESET_JACOBIAN(VFOCAL_CAM2, 2, 4);
    RESET_JACOBIAN(HCENTER_CAM1, 2, 4);
    RESET_JACOBIAN(HCENTER_CAM2, 2, 4);
    RESET_JACOBIAN(VCENTER_CAM1, 2, 4);
    RESET_JACOBIAN(VCENTER_CAM2, 2, 4);
    RESET_JACOBIAN(LENSDISTORTA_CAM1, 2, 4);
    RESET_JACOBIAN(LENSDISTORTA_CAM2, 2, 4);
    RESET_JACOBIAN(LENSDISTORTB_CAM1, 2, 4);
    RESET_JACOBIAN(LENSDISTORTB_CAM2, 2, 4);
    RESET_JACOBIAN(LENSDISTORTC_CAM1, 2, 4);
    RESET_JACOBIAN(LENSDISTORTC_CAM2, 2, 4);
    RESET_JACOBIAN(ROTATION_CAM1, 2, 9);
    RESET_JACOBIAN(ROTATION_CAM2, 2, 9);

#undef RESET_JACOBIAN
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    bool validProj, validLift;
    Eigen::Vector3d refpt;
    Eigen::Vector2d estimpt;
    Eigen::Matrix<double, 3, 4> Jhfocal_lift;
    Eigen::Matrix<double, 3, 4> Jvfocal_lift;
    Eigen::Matrix<double, 3, 4> Jhcenter_lift;
    Eigen::Matrix<double, 3, 4> Jvcenter_lift;
    Eigen::Matrix<double, 3, 4> JdistortA_lift;
    Eigen::Matrix<double, 3, 4> JdistortB_lift;
    Eigen::Matrix<double, 3, 4> JdistortC_lift;
    Eigen::Matrix<double, 3, 9> Jrotation_lift;
    Eigen::Matrix<double, 3, 4> JtX_lift;
    Eigen::Matrix<double, 3, 4> JtY_lift;
    Eigen::Matrix<double, 3, 4> JtZ_lift;
    Eigen::Matrix<double, 2, 3> Jpoint_project;
    Eigen::Matrix<double, 2, 4> Jhfocal_project;
    Eigen::Matrix<double, 2, 4> Jvfocal_project;
    Eigen::Matrix<double, 2, 4> Jhcenter_project;
    Eigen::Matrix<double, 2, 4> Jvcenter_project;
    Eigen::Matrix<double, 2, 4> JdistortA_project;
    Eigen::Matrix<double, 2, 4> JdistortB_project;
    Eigen::Matrix<double, 2, 4> JdistortC_project;
    Eigen::Matrix<double, 2, 9> Jrotation_project;
    Eigen::Matrix<double, 2, 4> JtX_project;
    Eigen::Matrix<double, 2, 4> JtY_project;
    Eigen::Matrix<double, 2, 4> JtZ_project;

    residuals[0] = 0.0;
    residuals[1] = 0.0;
    resetJacobians(jacobians);

    if (has(INDEX_HFOCAL_CAM1)) {
      camera1->setHorizontalFocal(parameters[map(INDEX_HFOCAL_CAM1)]);
    }
    if (has(INDEX_HFOCAL_CAM2)) {
      camera2->setHorizontalFocal(parameters[map(INDEX_HFOCAL_CAM2)]);
    } else {
      camera2->setHorizontalFocal(parameters[map(INDEX_HFOCAL_CAM1)]);
    }
    if (has(INDEX_VFOCAL_CAM1)) {
      camera1->setVerticalFocal(parameters[map(INDEX_VFOCAL_CAM1)]);
    }
    if (has(INDEX_VFOCAL_CAM2)) {
      camera2->setVerticalFocal(parameters[map(INDEX_VFOCAL_CAM2)]);
    } else {
      camera2->setVerticalFocal(parameters[map(INDEX_VFOCAL_CAM1)]);
    }
    if (has(INDEX_HCENTER_CAM1)) {
      camera1->setHorizontalCenter(parameters[map(INDEX_HCENTER_CAM1)]);
    }
    if (has(INDEX_HCENTER_CAM2)) {
      camera2->setHorizontalCenter(parameters[map(INDEX_HCENTER_CAM2)]);
    }
    if (has(INDEX_VCENTER_CAM1)) {
      camera1->setVerticalCenter(parameters[map(INDEX_VCENTER_CAM1)]);
    }
    if (has(INDEX_VCENTER_CAM2)) {
      camera2->setVerticalCenter(parameters[map(INDEX_VCENTER_CAM2)]);
    }
    if (has(INDEX_LENSDISTORTA_CAM1)) {
      camera1->setDistortionA(parameters[map(INDEX_LENSDISTORTA_CAM1)]);
    }
    if (has(INDEX_LENSDISTORTA_CAM2)) {
      camera2->setDistortionA(parameters[map(INDEX_LENSDISTORTA_CAM2)]);
    }
    if (has(INDEX_LENSDISTORTB_CAM1)) {
      camera1->setDistortionB(parameters[map(INDEX_LENSDISTORTB_CAM1)]);
    }
    if (has(INDEX_LENSDISTORTB_CAM2)) {
      camera2->setDistortionB(parameters[map(INDEX_LENSDISTORTB_CAM2)]);
    }
    if (has(INDEX_LENSDISTORTC_CAM1)) {
      camera1->setDistortionC(parameters[map(INDEX_LENSDISTORTC_CAM1)]);
    }
    if (has(INDEX_LENSDISTORTC_CAM2)) {
      camera2->setDistortionC(parameters[map(INDEX_LENSDISTORTC_CAM2)]);
    }
    if (has(INDEX_ROTATION_CAM1)) {
      camera1->setRotation(parameters[map(INDEX_ROTATION_CAM1)]);
    }
    if (has(INDEX_ROTATION_CAM2)) {
      camera2->setRotation(parameters[map(INDEX_ROTATION_CAM2)]);
    }

    validLift =
        camera1->lift(refpt, Jhfocal_lift, Jvfocal_lift, Jhcenter_lift, Jvcenter_lift, JdistortA_lift, JdistortB_lift,
                      JdistortC_lift, Jrotation_lift, JtX_lift, JtY_lift, JtZ_lift, impt1, sphereScale);
    if (!validLift) {
      return true;
    }

    assert(std::abs(refpt.norm() - sphereScale) < 1e-6 && (bool)"lift() failed to put point at sphereScale");

    validProj = camera2->project(estimpt, Jpoint_project, Jhfocal_project, Jvfocal_project, Jhcenter_project,
                                 Jvcenter_project, JdistortA_project, JdistortB_project, JdistortC_project,
                                 Jrotation_project, JtX_project, JtY_project, JtZ_project, refpt);
    if (!validProj) {
      return true;
    }

    residuals[0] = estimpt(0) - impt2(0);
    residuals[1] = estimpt(1) - impt2(1);

    if (VS_ISNAN(residuals[0])) {
      residuals[0] = 0;
      residuals[1] = 0;
      return true;
    }

    if (jacobians == nullptr) {
      return true;
    }

#define SET_JACOBIAN(PARAM, ROWS, COLS, VALUE)                                                        \
  if (has(INDEX_##PARAM) && jacobians[map(INDEX_##PARAM)] != nullptr) {                               \
    Eigen::Map<Eigen::Matrix<double, ROWS, COLS, Eigen::RowMajor> > J(jacobians[map(INDEX_##PARAM)]); \
    J = (VALUE);                                                                                      \
  }

    SET_JACOBIAN(HFOCAL_CAM1, 2, 4, Jpoint_project * Jhfocal_lift);
    SET_JACOBIAN(HFOCAL_CAM2, 2, 4, Jhfocal_project);
    SET_JACOBIAN(VFOCAL_CAM1, 2, 4, Jpoint_project * Jvfocal_lift);
    SET_JACOBIAN(VFOCAL_CAM2, 2, 4, Jvfocal_project);
    SET_JACOBIAN(HCENTER_CAM1, 2, 4, Jpoint_project * Jhcenter_lift);
    SET_JACOBIAN(HCENTER_CAM2, 2, 4, Jhcenter_project);
    SET_JACOBIAN(VCENTER_CAM1, 2, 4, Jpoint_project * Jvcenter_lift);
    SET_JACOBIAN(VCENTER_CAM2, 2, 4, Jvcenter_project);
    SET_JACOBIAN(LENSDISTORTA_CAM1, 2, 4, Jpoint_project * JdistortA_lift);
    SET_JACOBIAN(LENSDISTORTA_CAM2, 2, 4, JdistortA_project);
    SET_JACOBIAN(LENSDISTORTB_CAM1, 2, 4, Jpoint_project * JdistortB_lift);
    SET_JACOBIAN(LENSDISTORTB_CAM2, 2, 4, JdistortB_project);
    SET_JACOBIAN(LENSDISTORTC_CAM1, 2, 4, Jpoint_project * JdistortC_lift);
    SET_JACOBIAN(LENSDISTORTC_CAM2, 2, 4, JdistortC_project);
    SET_JACOBIAN(ROTATION_CAM1, 2, 9, Jpoint_project * Jrotation_lift);
    SET_JACOBIAN(ROTATION_CAM2, 2, 9, Jrotation_project);

#undef SET_JACOBIAN

    return true;
  }

 private:
  std::shared_ptr<Camera> camera1;
  std::shared_ptr<Camera> camera2;
  Eigen::Vector2d impt1;
  Eigen::Vector2d impt2;
  double sphereScale;
  bool needsScaling;
  std::unordered_map<char, size_t> hashmap;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
