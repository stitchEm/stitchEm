// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CERESLIB_UNSUPPORTED

#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif  // GLOG_NO_ABBREVIATED_SEVERITIES

#include "calibrationRefinement.hpp"
#include "camera_fisheye.hpp"
#include "camera_nextfisheye.hpp"
#include "camera_perspective.hpp"
#include "so3Parameterization.hpp"
#include "boundedParameterization.hpp"
#include "inputDistance.hpp"
#include "rotationDistance.hpp"
#include "eigengeometry.hpp"
#include "calibrationUtils.hpp"

#include "common/angles.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/controlPointListDef.hpp"
#include "libvideostitch/logging.hpp"

#include <ceres/ceres.h>

namespace VideoStitch {
namespace Calibration {

CalibrationRefinement::CalibrationRefinement() {}

CalibrationRefinement::~CalibrationRefinement() {}

std::shared_ptr<Camera> CalibrationRefinement::getCamera(size_t index) {
  if (index > cameras.size()) {
    return std::shared_ptr<Camera>(nullptr);
  }
  return cameras[index];
}

void CalibrationRefinement::setupWithCameras(const std::vector<std::shared_ptr<Camera> >& cameras) {
  this->cameras = cameras;
}

Status CalibrationRefinement::process(
    Core::PanoDefinition& pano,
    const std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& filteredCPMap,
    const CalibrationConfig& calibrationConfig) {
  ceres::Problem problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  // do we need to estimate a single focal across cameras ?
  if (calibrationConfig.hasSingleFocal()) {
    // reuse the focal of first camera for the other ones
    for (size_t i = 1; i < cameras.size(); i++) {
      cameras[i]->tieFocalTo(*cameras[0]);
    }

    double* ptr = cameras[0]->getHorizontalFocalPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);

    ptr = cameras[0]->getVerticalFocalPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  } else {
    for (size_t i = 0; i < cameras.size(); i++) {
      double* ptr = cameras[i]->getHorizontalFocalPtr();
      problem.AddParameterBlock(ptr, 4);
      problem.SetParameterization(ptr, new BoundedParameterization);
    }

    for (size_t i = 0; i < cameras.size(); i++) {
      double* ptr = cameras[i]->getVerticalFocalPtr();
      problem.AddParameterBlock(ptr, 4);
      problem.SetParameterization(ptr, new BoundedParameterization);
    }
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getHorizontalCenterPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getVerticalCenterPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getDistortionAPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getDistortionBPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getDistortionCPtr();
    problem.AddParameterBlock(ptr, 4);
    problem.SetParameterization(ptr, new BoundedParameterization);
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getRotationPtr();
    problem.AddParameterBlock(ptr, 9);
    problem.SetParameterization(ptr, new SO3Parameterization);
    // keep the camera 0 orientation constant
    if (i == 0) {
      problem.SetParameterBlockConstant(ptr);
    }
  }

  // TODO FIXMELATER no parameter for the translations yet
  // will be added when translations are estimated by the calibration algorithm

  // count number of control points in map
  size_t nControlPoints = 0;
  for (const auto& map : filteredCPMap) {
    nControlPoints += map.second.size();
  }

  for (size_t i = 0; i < cameras.size(); i++) {
    double* ptr = cameras[i]->getRotationPtr();

    // rotation distance checking the rotation is within presets
    rotationDistanceCostFunction* rotfunc = new rotationDistanceCostFunction(cameras[i]);
    problem.AddResidualBlock(rotfunc, nullptr, ptr);

    // should the rotation be constant ?
    if (cameras[i]->isRotationConstant()) {
      // declare constant rotation to ceres
      problem.SetParameterBlockConstant(ptr);

      // reset the rotation estimated for the extrinsics to the one from presets
      cameras[i]->setRotationFromPresets();
    } else {
      // if rotation estimated for the extrinsics is not within presets, set it to the presets
      // otherwise the rotation distance will always return false
      if (!cameras[i]->isRotationWithinPresets(cameras[i]->getRotation())) {
        Logger::get(Logger::Warning) << "Rotation estimated from extrinsics is out-of-presets, resetting it"
                                     << std::endl;
        cameras[i]->setRotationFromPresets();
      }
    }
  }

  // go through the ControlPointList map
  for (const auto& map : filteredCPMap) {
    // go through the ControlPoints
    for (const Core::ControlPoint& cp : map.second) {
      std::shared_ptr<Camera> cam1 = cameras[cp.index0];
      std::shared_ptr<Camera> cam2 = cameras[cp.index1];

      double* c1hfp = cam1->getHorizontalFocalPtr();
      double* c2hfp = cam2->getHorizontalFocalPtr();
      double* c1vfp = cam1->getVerticalFocalPtr();
      double* c2vfp = cam2->getVerticalFocalPtr();
      double* c1hcp = cam1->getHorizontalCenterPtr();
      double* c2hcp = cam2->getHorizontalCenterPtr();
      double* c1vcp = cam1->getVerticalCenterPtr();
      double* c2vcp = cam2->getVerticalCenterPtr();
      double* c1dpa = cam1->getDistortionAPtr();
      double* c2dpa = cam2->getDistortionAPtr();
      double* c1dpb = cam1->getDistortionBPtr();
      double* c2dpb = cam2->getDistortionBPtr();
      double* c1dpc = cam1->getDistortionCPtr();
      double* c2dpc = cam2->getDistortionCPtr();
      double* c1pp = cam1->getRotationPtr();
      double* c2pp = cam2->getRotationPtr();

      Eigen::Vector2d pt1;
      pt1(0) = cp.x0;
      pt1(1) = cp.y0;

      Eigen::Vector2d pt2;
      pt2(0) = cp.x1;
      pt2(1) = cp.y1;

      inputDistanceCostFunction* costfunc;

      std::vector<double*> parameter_blocks_1;
      parameter_blocks_1.push_back(c1hfp);
      if (!calibrationConfig.hasSingleFocal()) {
        parameter_blocks_1.push_back(c2hfp);
      }
      parameter_blocks_1.push_back(c1vfp);
      if (!calibrationConfig.hasSingleFocal()) {
        parameter_blocks_1.push_back(c2vfp);
      }
      parameter_blocks_1.push_back(c1hcp);
      parameter_blocks_1.push_back(c2hcp);
      parameter_blocks_1.push_back(c1vcp);
      parameter_blocks_1.push_back(c2vcp);
      parameter_blocks_1.push_back(c1dpa);
      parameter_blocks_1.push_back(c2dpa);
      parameter_blocks_1.push_back(c1dpb);
      parameter_blocks_1.push_back(c2dpb);
      parameter_blocks_1.push_back(c1dpc);
      parameter_blocks_1.push_back(c2dpc);
      parameter_blocks_1.push_back(c1pp);
      parameter_blocks_1.push_back(c2pp);
      costfunc = new inputDistanceCostFunction(cam1, cam2, pt1, pt2, calibrationConfig, pano.getSphereScale());
      problem.AddResidualBlock(costfunc, nullptr, parameter_blocks_1);

      /**Symmetry*/
      std::vector<double*> parameter_blocks_2;
      parameter_blocks_2.push_back(c2hfp);
      if (!calibrationConfig.hasSingleFocal()) {
        parameter_blocks_2.push_back(c1hfp);
      }
      parameter_blocks_2.push_back(c2vfp);
      if (!calibrationConfig.hasSingleFocal()) {
        parameter_blocks_2.push_back(c1vfp);
      }
      parameter_blocks_2.push_back(c2hcp);
      parameter_blocks_2.push_back(c1hcp);
      parameter_blocks_2.push_back(c2vcp);
      parameter_blocks_2.push_back(c1vcp);
      parameter_blocks_2.push_back(c2dpa);
      parameter_blocks_2.push_back(c1dpa);
      parameter_blocks_2.push_back(c2dpb);
      parameter_blocks_2.push_back(c1dpb);
      parameter_blocks_2.push_back(c2dpc);
      parameter_blocks_2.push_back(c1dpc);
      parameter_blocks_2.push_back(c2pp);
      parameter_blocks_2.push_back(c1pp);
      costfunc = new inputDistanceCostFunction(cam2, cam1, pt2, pt1, calibrationConfig, pano.getSphereScale());
      problem.AddResidualBlock(costfunc, nullptr, parameter_blocks_2);
    }
  }

  if (nControlPoints == 0) {
    pano.setCalibrationCost(std::numeric_limits<double>::max());
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure, "Did not find any control point"};
  }

  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.num_threads = 1;
  options.minimizer_progress_to_stdout = (Logger::getLevel() >= Logger::Debug);
  ceres::Solve(options, &problem, &summary);

  if (summary.termination_type != ceres::CONVERGENCE && summary.termination_type != ceres::NO_CONVERGENCE) {
    pano.setCalibrationCost(std::numeric_limits<double>::max());
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure, "Calibration did not converge"};
  }

  FAIL_RETURN(fillPano(pano, cameras));

  double avg_cost_per_point = summary.final_cost / static_cast<double>(nControlPoints);
  Logger::get(Logger::Verbose) << "Calib result: " << summary.initial_cost << " -> " << summary.final_cost << " "
                               << nControlPoints << "  average: " << avg_cost_per_point << std::endl;
  pano.setCalibrationCost(avg_cost_per_point - 10. * static_cast<double>(nControlPoints));
  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
#endif /* CERESLIB_UNSUPPORTED */
