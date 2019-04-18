// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __clang_analyzer__  // VSA-7040

#include "camera.hpp"

#include "eigengeometry.hpp"
#include "jacobians.hpp"

#include "backend/cpp/core/transformStack.hpp"

#include "common/angles.hpp"
#include "core/radial.hpp"

#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Calibration {

Camera::Camera()
    : cameraR(cameraR_data.data()), format(Core::InputDefinition::Format::FullFrameFisheye), width(0), height(0) {
  yprReference.setZero();
  yprCovariance.setZero();
  cameraRreference.setIdentity();
  cameraRreference_covariance.setIdentity();
  cameraR.setIdentity();
}

Camera::~Camera() {}

Core::InputDefinition::Format Camera::getFormat() { return format; }

void Camera::setFormat(Core::InputDefinition::Format format) { this->format = format; }

void Camera::tieFocalTo(const Camera& other) {
  horizontal_focal.tieTo(other.horizontal_focal);
  vertical_focal.tieTo(other.vertical_focal);
}

void Camera::untieFocal() {
  horizontal_focal.untie();
  vertical_focal.untie();
}

void Camera::setupWithRigCameraDefinition(Core::RigCameraDefinition& rigcamdef) {
  Core::NormalDouble val;
  double minbound, maxbound;
  width = (int)rigcamdef.getCamera()->getWidth();
  height = (int)rigcamdef.getCamera()->getHeight();

#define SETBOUNDEDVALUEFROMCAMERADEF(EXPORT_NAME, INTERNAL_NAME, MIN_BOUND, MAX_BOUND) \
  val = rigcamdef.getCamera()->get##EXPORT_NAME();                                     \
  minbound = val.mean - 3 * std::sqrt(val.variance);                                   \
  maxbound = val.mean + 3 * std::sqrt(val.variance);                                   \
  if (minbound < MIN_BOUND) minbound = MIN_BOUND;                                      \
  if (maxbound > MAX_BOUND) maxbound = MAX_BOUND;                                      \
  INTERNAL_NAME.setBounds(minbound, maxbound);                                         \
  INTERNAL_NAME.setValue(val.mean);

  SETBOUNDEDVALUEFROMCAMERADEF(DistortionA, distort_A, -10000.0, 10000.0);
  SETBOUNDEDVALUEFROMCAMERADEF(DistortionB, distort_B, -10000.0, 10000.0);
  SETBOUNDEDVALUEFROMCAMERADEF(DistortionC, distort_C, -10000.0, 10000.0);

  SETBOUNDEDVALUEFROMCAMERADEF(Fu, horizontal_focal, 0.1, 10000.0);
  SETBOUNDEDVALUEFROMCAMERADEF(Fv, vertical_focal, 0.1, 10000.0);

  SETBOUNDEDVALUEFROMCAMERADEF(Cu, horizontal_center, 0., (double)width);
  SETBOUNDEDVALUEFROMCAMERADEF(Cv, vertical_center, 0., (double)height);

  /*Build pose*/
  double yaw = rigcamdef.getYawRadians().mean;
  double pitch = rigcamdef.getPitchRadians().mean;
  double roll = rigcamdef.getRollRadians().mean;
  /*Keep yaw/pitch/roll reference*/
  yprReference << yaw, pitch, roll;

  Eigen::Matrix3d R;
  rotationFromEulerZXY(R, yaw, pitch, roll);
  cameraRreference = R;
  cameraR = R;

  /*Build rotation covariance matrix*/
  Eigen::Matrix<double, 9, 3> Jypr;
  double var;
  yprCovariance.fill(0);
  var = rigcamdef.getYawRadians().variance;
  if (var > 4.0 * M_PI * M_PI) {
    var = 4.0 * M_PI * M_PI;
  }
  yprCovariance(0, 0) = var;
  var = rigcamdef.getPitchRadians().variance;
  if (var > 4.0 * M_PI * M_PI) {
    var = 4.0 * M_PI * M_PI;
  }
  yprCovariance(1, 1) = var;
  var = rigcamdef.getRollRadians().variance;
  if (var > 4.0 * M_PI * M_PI) {
    var = 4.0 * M_PI * M_PI;
  }
  yprCovariance(2, 2) = var;

  getJacobianRotationWrtYawPitchRoll(Jypr, yaw, pitch, roll);
  cameraRreference_covariance = Jypr * yprCovariance * Jypr.transpose();

#define SETBOUNDEDVALUE(EXPORT_NAME, INTERNAL_NAME, MIN_BOUND, MAX_BOUND) \
  val = rigcamdef.get##EXPORT_NAME();                                     \
  minbound = val.mean - 3 * std::sqrt(val.variance);                      \
  maxbound = val.mean + 3 * std::sqrt(val.variance);                      \
  if (minbound < MIN_BOUND) minbound = MIN_BOUND;                         \
  if (maxbound > MAX_BOUND) maxbound = MAX_BOUND;                         \
  INTERNAL_NAME.setBounds(minbound, maxbound);                            \
  INTERNAL_NAME.setValue(val.mean);

  SETBOUNDEDVALUE(TranslationX, t_x, -1000000.0, 1000000.0);
  SETBOUNDEDVALUE(TranslationY, t_y, -1000000.0, 1000000.0);
  SETBOUNDEDVALUE(TranslationZ, t_z, -1000000.0, 1000000.0);

#undef SETBOUNDEDVALUEFROMCAMERADEF
#undef SETBOUNDEDVALUE
}

bool Camera::lift(Eigen::Vector3d& refpt, Eigen::Matrix<double, 3, 4>& Jhfocal, Eigen::Matrix<double, 3, 4>& Jvfocal,
                  Eigen::Matrix<double, 3, 4>& Jhcenter, Eigen::Matrix<double, 3, 4>& Jvcenter,
                  Eigen::Matrix<double, 3, 4>& JdistortA, Eigen::Matrix<double, 3, 4>& JdistortB,
                  Eigen::Matrix<double, 3, 4>& JdistortC, Eigen::Matrix<double, 3, 9>& Jrotation,
                  Eigen::Matrix<double, 3, 4>& JtX, Eigen::Matrix<double, 3, 4>& JtY, Eigen::Matrix<double, 3, 4>& JtZ,
                  const Eigen::Vector2d& impt, const double sphereScale) {
  Eigen::Matrix<double, 3, 1> Jhfocalub;
  Eigen::Matrix<double, 3, 1> Jvfocalub;
  Eigen::Matrix<double, 3, 1> Jhcenterub;
  Eigen::Matrix<double, 3, 1> Jvcenterub;
  Eigen::Matrix<double, 3, 1> JdistortAub;
  Eigen::Matrix<double, 3, 1> JdistortBub;
  Eigen::Matrix<double, 3, 1> JdistortCub;
  Eigen::Matrix<double, 3, 1> JtXub;
  Eigen::Matrix<double, 3, 1> JtYub;
  Eigen::Matrix<double, 3, 1> JtZub;

  bool res = lift_unbounded(refpt, Jhfocalub, Jvfocalub, Jhcenterub, Jvcenterub, JdistortAub, JdistortBub, JdistortCub,
                            Jrotation, JtXub, JtYub, JtZub, impt, sphereScale);
  if (!res) {
    return false;
  }

  Eigen::Matrix<double, 1, 4> Jmatrix;
  horizontal_focal.getJacobian(Jmatrix);
  Jhfocal = Jhfocalub * Jmatrix;

  vertical_focal.getJacobian(Jmatrix);
  Jvfocal = Jvfocalub * Jmatrix;

  horizontal_center.getJacobian(Jmatrix);
  Jhcenter = Jhcenterub * Jmatrix;

  vertical_center.getJacobian(Jmatrix);
  Jvcenter = Jvcenterub * Jmatrix;

  distort_A.getJacobian(Jmatrix);
  JdistortA = JdistortAub * Jmatrix;

  distort_B.getJacobian(Jmatrix);
  JdistortB = JdistortBub * Jmatrix;

  distort_C.getJacobian(Jmatrix);
  JdistortC = JdistortCub * Jmatrix;

  t_x.getJacobian(Jmatrix);
  JtX = JtXub * Jmatrix;

  t_y.getJacobian(Jmatrix);
  JtY = JtYub * Jmatrix;

  t_z.getJacobian(Jmatrix);
  JtZ = JtZub * Jmatrix;

  return true;
}

bool Camera::lift_unbounded(Eigen::Vector3d& refpt, Eigen::Matrix<double, 3, 1>& Jhfocal,
                            Eigen::Matrix<double, 3, 1>& Jvfocal, Eigen::Matrix<double, 3, 1>& Jhcenter,
                            Eigen::Matrix<double, 3, 1>& Jvcenter, Eigen::Matrix<double, 3, 1>& JdistortA,
                            Eigen::Matrix<double, 3, 1>& JdistortB, Eigen::Matrix<double, 3, 1>& JdistortC,
                            Eigen::Matrix<double, 3, 9>& Jrotation, Eigen::Matrix<double, 3, 1>& JtX,
                            Eigen::Matrix<double, 3, 1>& JtY, Eigen::Matrix<double, 3, 1>& JtZ,
                            const Eigen::Vector2d& impt, const double sphereScale) {
  bool res;
  Eigen::Vector2d impt_meters;

  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();

  /*From pixel to meters*/
  impt_meters(0) = (impt(0) - cu) / fu;
  impt_meters(1) = (impt(1) - cv) / fv;

  /*Horizontal focal jacobian*/
  Eigen::Matrix<double, 2, 1> J_meters_wrt_hfocal;
  J_meters_wrt_hfocal(0, 0) = (cu - impt(0)) / (fu * fu);
  J_meters_wrt_hfocal(1, 0) = 0;

  /*Vertical focal jacobian*/
  Eigen::Matrix<double, 2, 1> J_meters_wrt_vfocal;
  J_meters_wrt_vfocal(0, 0) = 0;
  J_meters_wrt_vfocal(1, 0) = (cv - impt(1)) / (fv * fv);

  /*Horizontal center jacobian*/
  Eigen::Matrix<double, 2, 1> J_meters_wrt_hcenter;
  J_meters_wrt_hcenter(0, 0) = -1.0 / fu;
  J_meters_wrt_hcenter(1, 0) = 0.0;

  /*Vertical center jacobian*/
  Eigen::Matrix<double, 2, 1> J_meters_wrt_vcenter;
  J_meters_wrt_vcenter(0, 0) = 0.0;
  J_meters_wrt_vcenter(1, 0) = -1.0 / fv;

  /*Undistort*/
  Eigen::Vector2d impt_undistorted;
  Eigen::Matrix<double, 2, 2> J_undistorted_wrt_distorted;
  Eigen::Matrix<double, 2, 1> J_undistorted_wrt_distortion_parameter_A;
  Eigen::Matrix<double, 2, 1> J_undistorted_wrt_distortion_parameter_B;
  Eigen::Matrix<double, 2, 1> J_undistorted_wrt_distortion_parameter_C;
  res = undistort(impt_undistorted, J_undistorted_wrt_distorted, J_undistorted_wrt_distortion_parameter_A,
                  J_undistorted_wrt_distortion_parameter_B, J_undistorted_wrt_distortion_parameter_C, impt_meters);
  if (!res) {
    return false;
  }

  /*Back project*/
  Eigen::Vector3d campt;
  Eigen::Matrix<double, 3, 2> J_campt_wrt_impt_undistorted;
  res = backproject(campt, J_campt_wrt_impt_undistorted, impt_undistorted);
  if (!res) {
    return false;
  }

  /*Transform to reference space*/
  double tx = t_x.getValue();
  double ty = t_y.getValue();
  double tz = t_z.getValue();
  const Eigen::Vector3d translation(tx, ty, tz);

  Eigen::Matrix<double, 3, 3> J_refpt_wrt_campt;

  /*Set refpt to reference sphere scale by ray-tracing it, finding lambda > 0 such as norm(camcenter + lambda * (refpt -
   * camcenter)) == sphereScale*/
  if (translation.squaredNorm() > 1e-6) {
    /*cam_refpt is campt lifted to the camera unit sphere in reference space*/
    const Eigen::Vector3d cam_refpt = cameraR.transpose() * (campt - translation);
    const Eigen::Vector3d camcenterpt = -cameraR.transpose() * translation;
    const Eigen::Vector3d ray = cam_refpt - camcenterpt;

    /*lambda is root of lambda^2.ray^T.ray + 2 * lambda * center^T.ray  + center^T.center - sphereScale^2*/
    const double a = ray.squaredNorm();
    const double b = 2 * camcenterpt.dot(ray);
    const double c = camcenterpt.squaredNorm() - sphereScale * sphereScale;
    const double d = b * b - 4 * a * c;
    assert(d >= 0.);
    const double lambda = (d >= 0.) ? (-b + std::sqrt(d)) / (2 * a) : 1.;
    refpt = camcenterpt + lambda * ray;

    const Eigen::Matrix<double, 1, 3> J_a_wrt_cam_refpt = 2 * ray.transpose();
    const Eigen::Matrix<double, 1, 3> J_b_wrt_cam_refpt = 2 * camcenterpt.transpose();

    const double J_lambda_wrt_a =
        (d > 1e-6) ? -(-b * std::sqrt(d) - 2 * a * c + b * b) / (2 * a * a * std::sqrt(d)) : b / (2 * a * a);
    const double J_lambda_wrt_b = (d > 1e-6) ? -(std::sqrt(d) - b) / (2 * a * std::sqrt(d)) : -1. / (2 * a);

    const Eigen::Matrix<double, 3, 3> J_cam_refpt_wrt_campt = cameraR.transpose();
    const Eigen::Matrix<double, 1, 3> J_lambda_wrt_cam_refpt =
        J_lambda_wrt_a * J_a_wrt_cam_refpt + J_lambda_wrt_b * J_b_wrt_cam_refpt;
    const Eigen::Matrix<double, 1, 3> J_lambda_wrt_campt = J_lambda_wrt_cam_refpt * J_cam_refpt_wrt_campt;

    /*Jacobian of transform wrt campt*/
    J_refpt_wrt_campt = lambda * J_cam_refpt_wrt_campt + ray * J_lambda_wrt_campt;
  } else {
    /*Rotate campt to refpt and set to sphereScale*/
    refpt = sphereScale * (cameraR.transpose() * campt);
    /*Jacobian of transform wrt campt*/
    J_refpt_wrt_campt = sphereScale * cameraR.transpose();
  }

  /*Check correctness of scale*/
  assert(std::abs(refpt.norm() - sphereScale) < 1e-6);

  /*Jacobian of transform wrt rotation*/

  /*ref = (update * cRr)' * cam*/
  /*ref = M' * cam*/
  /*Jref = JMpCam_wrt_M*/

  Jrotation.fill(0);
  Jrotation(0, 0) = cameraR(0, 0) * campt(0);
  Jrotation(1, 0) = cameraR(0, 1) * campt(0);
  Jrotation(2, 0) = cameraR(0, 2) * campt(0);
  Jrotation(0, 1) = cameraR(0, 0) * campt(1);
  Jrotation(1, 1) = cameraR(0, 1) * campt(1);
  Jrotation(2, 1) = cameraR(0, 2) * campt(1);
  Jrotation(0, 2) = cameraR(0, 0) * campt(2);
  Jrotation(1, 2) = cameraR(0, 1) * campt(2);
  Jrotation(2, 2) = cameraR(0, 2) * campt(2);
  Jrotation(0, 3) = cameraR(1, 0) * campt(0);
  Jrotation(1, 3) = cameraR(1, 1) * campt(0);
  Jrotation(2, 3) = cameraR(1, 2) * campt(0);
  Jrotation(0, 4) = cameraR(1, 0) * campt(1);
  Jrotation(1, 4) = cameraR(1, 1) * campt(1);
  Jrotation(2, 4) = cameraR(1, 2) * campt(1);
  Jrotation(0, 5) = cameraR(1, 0) * campt(2);
  Jrotation(1, 5) = cameraR(1, 1) * campt(2);
  Jrotation(2, 5) = cameraR(1, 2) * campt(2);
  Jrotation(0, 6) = cameraR(2, 0) * campt(0);
  Jrotation(1, 6) = cameraR(2, 1) * campt(0);
  Jrotation(2, 6) = cameraR(2, 2) * campt(0);
  Jrotation(0, 7) = cameraR(2, 0) * campt(1);
  Jrotation(1, 7) = cameraR(2, 1) * campt(1);
  Jrotation(2, 7) = cameraR(2, 2) * campt(1);
  Jrotation(0, 8) = cameraR(2, 0) * campt(2);
  Jrotation(1, 8) = cameraR(2, 1) * campt(2);
  Jrotation(2, 8) = cameraR(2, 2) * campt(2);

  /*Jacobian of transform wrt translation*/
  JtX(0, 0) = -cameraR(0, 0);
  JtX(1, 0) = -cameraR(0, 1);
  JtX(2, 0) = -cameraR(0, 2);
  JtY(0, 0) = -cameraR(1, 0);
  JtY(1, 0) = -cameraR(1, 1);
  JtY(2, 0) = -cameraR(1, 2);
  JtZ(0, 0) = -cameraR(2, 0);
  JtZ(1, 0) = -cameraR(2, 1);
  JtZ(2, 0) = -cameraR(2, 2);

  JdistortA = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distortion_parameter_A;
  JdistortB = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distortion_parameter_B;
  JdistortC = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distortion_parameter_C;
  Jhfocal = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distorted * J_meters_wrt_hfocal;
  Jvfocal = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distorted * J_meters_wrt_vfocal;
  Jhcenter = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distorted * J_meters_wrt_hcenter;
  Jvcenter = J_refpt_wrt_campt * J_campt_wrt_impt_undistorted * J_undistorted_wrt_distorted * J_meters_wrt_vcenter;

  return true;
}

bool Camera::lift(Eigen::Vector3d& refpt, const Eigen::Vector2d& impt, const double sphereScale) {
  bool res;
  Eigen::Vector2d impt_meters;

  // static int count = 0;

  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();

  /*From pixel to meters*/
  impt_meters(0) = (impt(0) - cu) / fu;
  impt_meters(1) = (impt(1) - cv) / fv;

  /*Undistort*/
  Eigen::Vector2d impt_undistorted;
  res = undistort(impt_undistorted, impt_meters);
  if (!res) {
    return false;
  }

  /*Back project*/
  Eigen::Vector3d campt;
  Eigen::Matrix<double, 3, 2> J_campt_wrt_impt_undistorted;
  res = backproject(campt, J_campt_wrt_impt_undistorted, impt_undistorted);
  if (!res) {
    return false;
  }

  /*Transform to reference space*/
  double tx = t_x.getValue();
  double ty = t_y.getValue();
  double tz = t_z.getValue();
  Eigen::Vector3d translation(tx, ty, tz);
  refpt = cameraR.transpose() * (campt - translation);

  /*Set refpt to reference sphere scale by ray-tracing it, finding lambda > 0 such as norm(camcenter + lambda * (refpt -
   * camcenter)) == sphereScale*/
  if (translation.squaredNorm() > 1e-6) {
    /*cam_refpt is campt lifted to the camera unit sphere in reference space*/
    const Eigen::Vector3d cam_refpt = refpt;
    const Eigen::Vector3d camcenterpt = -cameraR.transpose() * translation;
    const Eigen::Vector3d ray = cam_refpt - camcenterpt;

    /*lambda is root of lambda^2.ray^T.ray + 2 * lambda * center^T.ray  + center^T.center - sphereScale^2*/
    const double a = ray.squaredNorm();
    const double b = 2 * camcenterpt.dot(ray);
    const double c = camcenterpt.squaredNorm() - sphereScale * sphereScale;
    const double d = b * b - 4 * a * c;
    assert(d >= 0.);
    const double lambda = (d >= 0.) ? (-b + std::sqrt(d)) / (2 * a) : 1.;
    refpt = camcenterpt + lambda * ray;
  } else {
    refpt *= sphereScale;
  }

  /*Check correctness of scale*/
  assert(std::abs(refpt.norm() - sphereScale) < 1e-6);

  return true;
}

bool Camera::quicklift(Eigen::Vector3d& campt, const Eigen::Vector2d& impt) {
  bool res;
  Eigen::Vector2d impt_meters;

  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();

  /*From pixel to meters*/
  impt_meters(0) = (impt(0) - cu) / fu;
  impt_meters(1) = (impt(1) - cv) / fv;

  /*Undistort*/
  Eigen::Vector2d impt_undistorted;
  res = undistort(impt_undistorted, impt_meters);
  if (!res) {
    return false;
  }

  /*Back project*/
  Eigen::Matrix<double, 3, 2> J_campt_wrt_impt_undistorted;
  res = backproject(campt, J_campt_wrt_impt_undistorted, impt_undistorted);
  if (!res) {
    return false;
  }

  return true;
}

bool Camera::project(Eigen::Vector2d& impt_pixels, Eigen::Matrix<double, 2, 3>& Jpoint,
                     Eigen::Matrix<double, 2, 4>& Jhfocal, Eigen::Matrix<double, 2, 4>& Jvfocal,
                     Eigen::Matrix<double, 2, 4>& Jhcenter, Eigen::Matrix<double, 2, 4>& Jvcenter,
                     Eigen::Matrix<double, 2, 4>& JdistortA, Eigen::Matrix<double, 2, 4>& JdistortB,
                     Eigen::Matrix<double, 2, 4>& JdistortC, Eigen::Matrix<double, 2, 9>& Jrotation,
                     Eigen::Matrix<double, 2, 4>& JtX, Eigen::Matrix<double, 2, 4>& JtY,
                     Eigen::Matrix<double, 2, 4>& JtZ, const Eigen::Vector3d& refpt) {
  Eigen::Matrix<double, 2, 1> Jhfocalub;
  Eigen::Matrix<double, 2, 1> Jvfocalub;
  Eigen::Matrix<double, 2, 1> Jhcenterub;
  Eigen::Matrix<double, 2, 1> Jvcenterub;
  Eigen::Matrix<double, 2, 1> JdistortAub;
  Eigen::Matrix<double, 2, 1> JdistortBub;
  Eigen::Matrix<double, 2, 1> JdistortCub;
  Eigen::Matrix<double, 2, 1> JtXub;
  Eigen::Matrix<double, 2, 1> JtYub;
  Eigen::Matrix<double, 2, 1> JtZub;

  bool res = project_unbounded(impt_pixels, Jpoint, Jhfocalub, Jvfocalub, Jhcenterub, Jvcenterub, JdistortAub,
                               JdistortBub, JdistortCub, Jrotation, JtXub, JtYub, JtZub, refpt);
  if (!res) {
    return false;
  }

  Eigen::Matrix<double, 1, 4> Jmatrix;
  horizontal_focal.getJacobian(Jmatrix);
  Jhfocal = Jhfocalub * Jmatrix;

  vertical_focal.getJacobian(Jmatrix);
  Jvfocal = Jvfocalub * Jmatrix;

  horizontal_center.getJacobian(Jmatrix);
  Jhcenter = Jhcenterub * Jmatrix;

  vertical_center.getJacobian(Jmatrix);
  Jvcenter = Jvcenterub * Jmatrix;

  distort_A.getJacobian(Jmatrix);
  JdistortA = JdistortAub * Jmatrix;

  distort_B.getJacobian(Jmatrix);
  JdistortB = JdistortBub * Jmatrix;

  distort_C.getJacobian(Jmatrix);
  JdistortC = JdistortCub * Jmatrix;

  t_x.getJacobian(Jmatrix);
  JtX = JtXub * Jmatrix;

  t_y.getJacobian(Jmatrix);
  JtY = JtYub * Jmatrix;

  t_z.getJacobian(Jmatrix);
  JtZ = JtZub * Jmatrix;

  return true;
}

bool Camera::project_unbounded(Eigen::Vector2d& impt_pixels, Eigen::Matrix<double, 2, 3>& Jpoint,
                               Eigen::Matrix<double, 2, 1>& Jhfocal, Eigen::Matrix<double, 2, 1>& Jvfocal,
                               Eigen::Matrix<double, 2, 1>& Jhcenter, Eigen::Matrix<double, 2, 1>& Jvcenter,
                               Eigen::Matrix<double, 2, 1>& JdistortA, Eigen::Matrix<double, 2, 1>& JdistortB,
                               Eigen::Matrix<double, 2, 1>& JdistortC, Eigen::Matrix<double, 2, 9>& Jrotation,
                               Eigen::Matrix<double, 2, 1>& JtX, Eigen::Matrix<double, 2, 1>& JtY,
                               Eigen::Matrix<double, 2, 1>& JtZ, const Eigen::Vector3d& refpt) {
  /*From reference space to camera space*/
  Eigen::Vector3d campt;
  Eigen::Vector3d translation(t_x.getValue(), t_y.getValue(), t_z.getValue());
  campt = cameraR * refpt + translation;

  /*Jacobian of campt wrt refpt*/
  Eigen::Matrix<double, 3, 3> J_campt_wrt_refpt;
  J_campt_wrt_refpt = cameraR;

  /*Jacobian of campt wrt rotation*/
  Eigen::Matrix<double, 3, 9> J_campt_wrt_rotation;
  J_campt_wrt_rotation.fill(0);
  J_campt_wrt_rotation(0, 0) = campt(0);
  J_campt_wrt_rotation(1, 1) = campt(0);
  J_campt_wrt_rotation(2, 2) = campt(0);
  J_campt_wrt_rotation(0, 3) = campt(1);
  J_campt_wrt_rotation(1, 4) = campt(1);
  J_campt_wrt_rotation(2, 5) = campt(1);
  J_campt_wrt_rotation(0, 6) = campt(2);
  J_campt_wrt_rotation(1, 7) = campt(2);
  J_campt_wrt_rotation(2, 8) = campt(2);

  /*Jacobian of campt wrt translation*/
  Eigen::Matrix<double, 3, 1> J_campt_wrt_tX;
  Eigen::Matrix<double, 3, 1> J_campt_wrt_tY;
  Eigen::Matrix<double, 3, 1> J_campt_wrt_tZ;
  J_campt_wrt_tX(0, 0) = 1.;
  J_campt_wrt_tX(1, 0) = 0.;
  J_campt_wrt_tX(2, 0) = 0.;
  J_campt_wrt_tY(0, 0) = 0.;
  J_campt_wrt_tY(1, 0) = 1.;
  J_campt_wrt_tY(2, 0) = 0.;
  J_campt_wrt_tZ(0, 0) = 0.;
  J_campt_wrt_tZ(1, 0) = 0.;
  J_campt_wrt_tZ(2, 0) = 1.;

  /*Effective projection*/
  bool res;
  Eigen::Vector2d impt_meters;
  Eigen::Matrix<double, 2, 3> J_impt_meters_wrt_campt;
  res = project(impt_meters, J_impt_meters_wrt_campt, campt);
  if (!res) {
    return false;
  }

  /*Distortion*/
  Eigen::Vector2d impt_distorted;
  Eigen::Matrix<double, 2, 2> J_impt_distorted_wrt_impt_meters;
  Eigen::Matrix<double, 2, 1> J_impt_distorted_wrt_distortion_parameter_A;
  Eigen::Matrix<double, 2, 1> J_impt_distorted_wrt_distortion_parameter_B;
  Eigen::Matrix<double, 2, 1> J_impt_distorted_wrt_distortion_parameter_C;
  res = distort(impt_distorted, J_impt_distorted_wrt_impt_meters, J_impt_distorted_wrt_distortion_parameter_A,
                J_impt_distorted_wrt_distortion_parameter_B, J_impt_distorted_wrt_distortion_parameter_C, impt_meters);
  if (!res) {
    return false;
  }

  /*From meters to pixels*/
  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();
  impt_pixels(0) = fu * impt_distorted(0) + cu;
  impt_pixels(1) = fv * impt_distorted(1) + cv;

  /*Jacobians meters to pixels*/
  Eigen::Matrix<double, 2, 2> J_impt_pixels_wrt_impt_distorted;
  J_impt_pixels_wrt_impt_distorted.fill(0);
  J_impt_pixels_wrt_impt_distorted(0, 0) = fu;
  J_impt_pixels_wrt_impt_distorted(1, 1) = fv;

  Eigen::Matrix<double, 2, 1> J_impt_pixels_wrt_hfocal;
  J_impt_pixels_wrt_hfocal(0, 0) = impt_distorted(0);
  J_impt_pixels_wrt_hfocal(1, 0) = 0;

  Eigen::Matrix<double, 2, 1> J_impt_pixels_wrt_vfocal;
  J_impt_pixels_wrt_vfocal(0, 0) = 0;
  J_impt_pixels_wrt_vfocal(1, 0) = impt_distorted(1);

  Eigen::Matrix<double, 2, 1> J_impt_pixels_wrt_hcenter;
  J_impt_pixels_wrt_hcenter(0, 0) = 1.0;
  J_impt_pixels_wrt_hcenter(1, 0) = 0.0;

  Eigen::Matrix<double, 2, 1> J_impt_pixels_wrt_vcenter;
  J_impt_pixels_wrt_vcenter(0, 0) = 0.0;
  J_impt_pixels_wrt_vcenter(1, 0) = 1.0;

  Jpoint =
      J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_impt_meters * J_impt_meters_wrt_campt * J_campt_wrt_refpt;

  Jrotation = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_impt_meters * J_impt_meters_wrt_campt *
              J_campt_wrt_rotation;
  JtX = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_impt_meters * J_impt_meters_wrt_campt * J_campt_wrt_tX;
  JtY = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_impt_meters * J_impt_meters_wrt_campt * J_campt_wrt_tY;
  JtZ = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_impt_meters * J_impt_meters_wrt_campt * J_campt_wrt_tZ;
  Jhfocal = J_impt_pixels_wrt_hfocal;
  Jvfocal = J_impt_pixels_wrt_vfocal;
  Jhcenter = J_impt_pixels_wrt_hcenter;
  Jvcenter = J_impt_pixels_wrt_vcenter;
  JdistortA = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_distortion_parameter_A;
  JdistortB = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_distortion_parameter_B;
  JdistortC = J_impt_pixels_wrt_impt_distorted * J_impt_distorted_wrt_distortion_parameter_C;

  return true;
}

bool Camera::project(Eigen::Vector2d& impt_pixels, const Eigen::Vector3d& refpt) {
  /*From reference space to camera space*/
  Eigen::Vector3d campt;
  Eigen::Vector3d translation(t_x.getValue(), t_y.getValue(), t_z.getValue());
  campt = cameraR * refpt + translation;

  /*effective projection*/
  bool res;
  Eigen::Vector2d impt_meters;
  Eigen::Matrix<double, 2, 3> J_impt_meters_wrt_campt;
  res = project(impt_meters, J_impt_meters_wrt_campt, campt);
  if (!res) {
    return false;
  }

  /*Distortion*/
  Eigen::Vector2d impt_distorted;
  res = distort(impt_distorted, impt_meters);
  if (!res) {
    return false;
  }

  /*From meters to pixels*/
  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();
  impt_pixels(0) = fu * impt_distorted(0) + cu;
  impt_pixels(1) = fv * impt_distorted(1) + cv;

  return true;
}

bool Camera::quickproject(Eigen::Vector2d& impt_pixels, const Eigen::Vector3d& campt) {
  /*effective projection*/
  bool res;
  Eigen::Vector2d impt_meters;
  Eigen::Matrix<double, 2, 3> J_impt_meters_wrt_campt;
  res = project(impt_meters, J_impt_meters_wrt_campt, campt);
  if (!res) {
    return false;
  }

  /*Distortion*/
  Eigen::Vector2d impt_distorted;
  res = distort(impt_distorted, impt_meters);
  if (!res) {
    return false;
  }

  /*From meters to pixels*/
  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();
  impt_pixels(0) = fu * impt_distorted(0) + cu;
  impt_pixels(1) = fv * impt_distorted(1) + cv;

  return true;
}

bool Camera::undistort(Eigen::Vector2d& impt_undistorted, const Eigen::Vector2d& impt_distorted) {
  double A = distort_A.getValue();
  double B = distort_B.getValue();
  double C = distort_C.getValue();
  double D = 1.0 - (A + B + C);
  float radial3 = (float)A;
  float radial2 = (float)B;
  float radial1 = (float)C;
  float radial0 = (float)1.0f - (radial3 + radial2 + radial1);
  float radial4 = (float)Core::computeRadial4(D, C, B, A);

  /**
   Check if floating point precision is large enough to solve the undistortion, otherwise, return false
   */
  if (std::abs(D - 1.0) > std::numeric_limits<double>::epsilon() &&
      std::abs(radial0 - 1.0f) <= std::numeric_limits<float>::epsilon()) {
    // radial 1, radial2 and radial3 will not sum up to 1 - (radial3 + radial2 + radial1) due to numerical floating
    // point issues do not try to solve the undistortion
    return false;
  }

  float2 undistorted = Core::TransformStack::inverseDistortionScaled(
      {float(impt_distorted[0]), float(impt_distorted[1])}, {{radial0, radial1, radial2, radial3, radial4}});

  impt_undistorted[0] = undistorted.x;
  impt_undistorted[1] = undistorted.y;

  /**
   Check correctness of undistortion
   */

  /**
   Direct distortion :
   x' = x * f(len)
   y' = y * f(len)
   len = sqrt(x*x+y*y)
   f(len) = A*len^3 + B*len^2 + C*len + D
   */
  double distorted_length = impt_distorted.norm();
  double undistorted_length = impt_undistorted.norm();

  if (undistorted_length > radial4) {
    return false;
  }

  double undistorted_length2 = undistorted_length * undistorted_length;
  double undistorted_length3 = undistorted_length2 * undistorted_length;
  double f = (A * undistorted_length3 + B * undistorted_length2 + C * undistorted_length + D);

  /*Distortion model can fail*/
  if (fabs(f * undistorted_length - distorted_length) > 1e-2) {
    assert(false && "should not get here");
    return false;
  }

  return true;
}

bool Camera::undistort(Eigen::Vector2d& impt_undistorted, Eigen::Matrix<double, 2, 2>& Jundistortedwrtdistorted,
                       Eigen::Matrix<double, 2, 1>& JparameterA, Eigen::Matrix<double, 2, 1>& JparameterB,
                       Eigen::Matrix<double, 2, 1>& JparameterC, const Eigen::Vector2d& impt_distorted) {
  double A = distort_A.getValue();
  double B = distort_B.getValue();
  double C = distort_C.getValue();
  double D = 1.0 - (A + B + C);

  double x = impt_distorted(0);
  double y = impt_distorted(1);
  double distorted_length = impt_distorted.norm();

  if (fabs(distorted_length) < 1e-6) {
    impt_undistorted(0) = x;
    impt_undistorted(1) = y;
    Jundistortedwrtdistorted.setIdentity();
    JparameterA.fill(0);
    JparameterB.fill(0);
    JparameterC.fill(0);
    return true;
  }

  bool res = undistort(impt_undistorted, impt_distorted);
  if (!res) {
    return false;
  }

  /**
   Direct distortion :
   x' = x * f(len)
   y' = y * f(len)
   len = sqrt(x*x+y*y)
   f(len) = A*len^3 + B*len^2 + C*len + D
   */
  Eigen::Matrix<double, 2, 2> Jdistortedwrtundistorted;
  double undistorted_length = impt_undistorted.norm();
  double undistorted_length2 = undistorted_length * undistorted_length;
  double undistorted_length3 = undistorted_length2 * undistorted_length;
  double f = (A * undistorted_length3 + B * undistorted_length2 + C * undistorted_length + D);

  double dlengthdx = impt_undistorted(0) / undistorted_length;
  double dlengthdy = impt_undistorted(1) / undistorted_length;
  double dfdlen = undistorted_length * (3.0 * A * undistorted_length + 2.0 * B) + C;
  double dfdx = dfdlen * dlengthdx;
  double dfdy = dfdlen * dlengthdy;
  Jdistortedwrtundistorted(0, 0) = impt_undistorted(0) * dfdx + f * 1.0;
  Jdistortedwrtundistorted(0, 1) = impt_undistorted(0) * dfdy;
  Jdistortedwrtundistorted(1, 0) = impt_undistorted(1) * dfdx;
  Jdistortedwrtundistorted(1, 1) = impt_undistorted(1) * dfdy + f * 1.0;

  /*Were interested in the other way ...*/
  Jundistortedwrtdistorted = Jdistortedwrtundistorted.inverse();

  double dfda = undistorted_length3 - 1.0;
  double dfdb = undistorted_length2 - 1.0;
  double dfdc = undistorted_length - 1.0;

  /*Derivative wrt parameters**/
  JparameterA(0, 0) = -(x * dfda) / (f * f);
  JparameterA(1, 0) = -(y * dfda) / (f * f);
  JparameterB(0, 0) = -(x * dfdb) / (f * f);
  JparameterB(1, 0) = -(y * dfdb) / (f * f);
  JparameterC(0, 0) = -(x * dfdc) / (f * f);
  JparameterC(1, 0) = -(y * dfdc) / (f * f);

  return true;
}

bool Camera::distort(Eigen::Vector2d& impt_distorted, const Eigen::Vector2d& impt_undistorted) {
  double A = distort_A.getValue();
  double B = distort_B.getValue();
  double C = distort_C.getValue();
  double D = 1.0 - (A + B + C);

  double x = impt_undistorted(0);
  double y = impt_undistorted(1);
  double undistorted_length = impt_undistorted.norm();

  if (undistorted_length < 1e-6) {
    impt_distorted(0) = 0;
    impt_distorted(1) = 0;
    return true;
  }

  double limit_distortion = Core::computeRadial4(D, C, B, A);
  if (undistorted_length > limit_distortion) {
    return false;
  }

  double undistorted_length2 = undistorted_length * undistorted_length;
  double undistorted_length3 = undistorted_length2 * undistorted_length;
  double distorted_length = A * undistorted_length3 + B * undistorted_length2 + C * undistorted_length + D;
  impt_distorted(0) = x * distorted_length;
  impt_distorted(1) = y * distorted_length;

  return true;
}

bool Camera::distort(Eigen::Vector2d& impt_distorted, Eigen::Matrix<double, 2, 2>& Jdistortedwrtundistorted,
                     Eigen::Matrix<double, 2, 1>& JparameterA, Eigen::Matrix<double, 2, 1>& JparameterB,
                     Eigen::Matrix<double, 2, 1>& JparameterC, const Eigen::Vector2d& impt_undistorted) {
  double A = distort_A.getValue();
  double B = distort_B.getValue();
  double C = distort_C.getValue();
  double D = 1.0 - (A + B + C);

  double x = impt_undistorted(0);
  double y = impt_undistorted(1);
  double undistorted_length = impt_undistorted.norm();

  if (undistorted_length < 1e-6) {
    impt_distorted(0) = 0;
    impt_distorted(1) = 0;
    Jdistortedwrtundistorted.setIdentity();
    JparameterA.fill(0);
    JparameterB.fill(0);
    JparameterC.fill(0);
    return true;
  }

  double limit_distortion = Core::computeRadial4(D, C, B, A);
  if (undistorted_length > limit_distortion) {
    return false;
  }

  double undistorted_length2 = undistorted_length * undistorted_length;
  double undistorted_length3 = undistorted_length2 * undistorted_length;
  double distorted_length = A * undistorted_length3 + B * undistorted_length2 + C * undistorted_length + D;
  impt_distorted(0) = x * distorted_length;
  impt_distorted(1) = y * distorted_length;

  double dundistorted_length_dx = x / undistorted_length;
  double dundistorted_length_dy = y / undistorted_length;
  double ddistorted_length_dundistorted_length = undistorted_length * (3.0 * A * undistorted_length + 2.0 * B) + C;

  double ddistorted_length_dx = ddistorted_length_dundistorted_length * dundistorted_length_dx;
  double ddistorted_length_dy = ddistorted_length_dundistorted_length * dundistorted_length_dy;

  Jdistortedwrtundistorted(0, 0) = 1.0 * distorted_length + x * ddistorted_length_dx;
  Jdistortedwrtundistorted(0, 1) = 0.0 * distorted_length + x * ddistorted_length_dy;
  Jdistortedwrtundistorted(1, 0) = 0.0 * distorted_length + y * ddistorted_length_dx;
  Jdistortedwrtundistorted(1, 1) = 1.0 * distorted_length + y * ddistorted_length_dy;

  double ddistorted_length_d_a = undistorted_length3 - 1.0;
  double ddistorted_length_d_b = undistorted_length2 - 1.0;
  double ddistorted_length_d_c = undistorted_length - 1.0;

  JparameterA(0, 0) = x * ddistorted_length_d_a;
  JparameterA(1, 0) = y * ddistorted_length_d_a;
  JparameterB(0, 0) = x * ddistorted_length_d_b;
  JparameterB(1, 0) = y * ddistorted_length_d_b;
  JparameterC(0, 0) = x * ddistorted_length_d_c;
  JparameterC(1, 0) = y * ddistorted_length_d_c;

  return true;
}

#define GENPARAMMEMBERS(PARAM_NAME, INTERNAL_PARAM_NAME)                                     \
                                                                                             \
  bool Camera::is##PARAM_NAME##Constant() const { return INTERNAL_PARAM_NAME.isConstant(); } \
                                                                                             \
  double* Camera::get##PARAM_NAME##Ptr() { return INTERNAL_PARAM_NAME.getMinimizerPtr(); }   \
                                                                                             \
  void Camera::set##PARAM_NAME(const double* ptr) { INTERNAL_PARAM_NAME.setMinimizerValues(ptr); }

GENPARAMMEMBERS(HorizontalFocal, horizontal_focal)
GENPARAMMEMBERS(VerticalFocal, vertical_focal)
GENPARAMMEMBERS(HorizontalCenter, horizontal_center)
GENPARAMMEMBERS(VerticalCenter, vertical_center)
GENPARAMMEMBERS(DistortionA, distort_A)
GENPARAMMEMBERS(DistortionB, distort_B)
GENPARAMMEMBERS(DistortionC, distort_C)
GENPARAMMEMBERS(TranslationX, t_x)
GENPARAMMEMBERS(TranslationY, t_y)
GENPARAMMEMBERS(TranslationZ, t_z)

#undef GENPARAMMEMBERS

bool Camera::isRotationConstant() const {
  // Rotation is constant if variances of yaw/pitch/roll are zero
  return (yprCovariance.trace() < 1e-6);
}

double* Camera::getRotationPtr() { return cameraR.data(); }

void Camera::setRotation(const double* ptr) {
  for (int i = 0; i < 9; i++) {
    cameraR_data[i] = ptr[i];
  }
}

Eigen::Matrix3d Camera::getRotation() const { return cameraR; }

void Camera::setRotationMatrix(const Eigen::Matrix3d& R) { cameraR = R; }

void Camera::setRotationFromPresets() { cameraR = cameraRreference; }

Eigen::Matrix3d Camera::getRotationFromPresets() const { return cameraRreference; }

bool Camera::isRotationWithinPresets(const Eigen::Matrix3d& R) const {
  Eigen::Vector3d yprDifference, ypr;

  EulerZXYFromRotation(ypr, R);
  yprDifference = ypr - yprReference;

  // handle the case when one angle is 177° and the other is -177°, the difference is actually -6 degrees, not 354
  for (int i = 0; i < ypr.size(); ++i) {
    if (yprDifference(i) > M_PI) {
      yprDifference(i) -= 2 * M_PI;
    } else if (yprDifference(i) <= -M_PI) {
      yprDifference(i) += 2 * M_PI;
    }
  }

  // return true if every difference angle is below 3*stddev, i.e.
  // angle^2 < 9*variance
  for (int i = 0; i < 3; i++) {
    if (yprDifference[i] * yprDifference[i] > 9 * yprCovariance(i, i)) {
      return false;
    }
  }
  return true;
}

Eigen::Vector3d Camera::getTranslation() const {
  return Eigen::Vector3d(t_x.getValue(), t_y.getValue(), t_z.getValue());
}

void Camera::setTranslation(const Eigen::Vector3d& translation) {
  t_x.setValue(translation(0));
  t_y.setValue(translation(1));
  t_z.setValue(translation(2));
}

void Camera::fillGeometry(Core::GeometryDefinition& g, int width, int height) {
  double fu = horizontal_focal.getValue();
  double fv = vertical_focal.getValue();
  double cu = horizontal_center.getValue();
  double cv = vertical_center.getValue();

  g.setHorizontalFocal(fu);
  if (fabs(fu - fv) > 1e-6) {
    g.setVerticalFocal(fv);
  }
  g.setCenterX(cu - ((double)width) / 2.0);
  g.setCenterY(cv - ((double)height) / 2.0);

  g.setDistortA(distort_A.getValue());
  g.setDistortB(distort_B.getValue());
  g.setDistortC(distort_C.getValue());

  // non-radial distortion not supported yet
  g.setDistortP1(0.0);
  g.setDistortP2(0.0);
  g.setDistortS1(0.0);
  g.setDistortS2(0.0);
  g.setDistortS3(0.0);
  g.setDistortS4(0.0);
  g.setDistortTau1(0.0);
  g.setDistortTau2(0.0);

  Eigen::Matrix3d R = getRotation();
  Eigen::Vector3d vr;
  EulerZXYFromRotation(vr, R);
  g.setYaw(radToDeg(vr(0)));
  g.setPitch(radToDeg(vr(1)));
  g.setRoll(radToDeg(vr(2)));

  double tx = t_x.getValue();
  double ty = t_y.getValue();
  double tz = t_z.getValue();

  if (std::abs(tx) >= 1e-6 || std::abs(ty) >= 1e-6 || std::abs(tz) >= 1e-6) {
    g.setTranslationX(tx);
    g.setTranslationY(ty);
    g.setTranslationZ(tz);
  }
}

size_t Camera::getWidth() { return width; }

size_t Camera::getHeight() { return height; }

void Camera::getRotationCovarianceMatrix(Eigen::Matrix<double, 9, 9>& cov) const { cov = cameraRreference_covariance; }

void Camera::getYawPitchRollCovarianceMatrix(Eigen::Matrix<double, 3, 3>& cov) const { cov = yprCovariance; }

void Camera::getRelativeRotation(Eigen::Matrix3d& second_Rmean_first, Eigen::Matrix3d& second_axisAngleRCov_first,
                                 const Camera& first, const Camera& second) {
  /*
  log(second * first.Transpose)
  */
  Eigen::Matrix<double, 3, 9> Jlog;
  Eigen::Matrix<double, 3, 3> R;
  second_Rmean_first = second.cameraR * first.cameraR.transpose();

  Eigen::Matrix<double, 9, 9> Jfirst;
  Eigen::Matrix<double, 9, 9> Jsecond;
  computedABtdA(Jfirst, second.cameraR, first.cameraR);
  computedABtdB(Jsecond, second.cameraR, first.cameraR);
  getJacobianAxisAngleWrtRotation(Jlog, second_Rmean_first);

  Eigen::Matrix<double, 3, 9> J1 = (Eigen::Matrix<double, 3, 9>)(Jlog * Jfirst);
  Eigen::Matrix<double, 3, 9> J2 = (Eigen::Matrix<double, 3, 9>)(Jlog * Jsecond);
  second_axisAngleRCov_first = J1 * first.cameraRreference_covariance * J1.transpose() +
                               J2 * second.cameraRreference_covariance * J2.transpose();
}

bool Camera::getLiftCovariance(Eigen::Vector3d& mean, Eigen::Matrix3d& cov, const double hfocal_variance,
                               const double vfocal_variance, const double hcenter_variance,
                               const double vcenter_variance, const double distorta_variance,
                               const double distortb_variance, const double distortc_variance, const double tx_variance,
                               const double ty_variance, const double tz_variance, const Eigen::Vector2d& impt,
                               const double sphereScale) {
  bool res;

  Eigen::Matrix<double, 3, 1> Jhfocal;
  Eigen::Matrix<double, 3, 1> Jvfocal;
  Eigen::Matrix<double, 3, 1> Jhcenter;
  Eigen::Matrix<double, 3, 1> Jvcenter;
  Eigen::Matrix<double, 3, 1> JdistortA;
  Eigen::Matrix<double, 3, 1> JdistortB;
  Eigen::Matrix<double, 3, 1> JdistortC;
  Eigen::Matrix<double, 3, 9> Jrotation;
  Eigen::Matrix<double, 3, 1> JtX;
  Eigen::Matrix<double, 3, 1> JtY;
  Eigen::Matrix<double, 3, 1> JtZ;

  res = lift_unbounded(mean, Jhfocal, Jvfocal, Jhcenter, Jvcenter, JdistortA, JdistortB, JdistortC, Jrotation, JtX, JtY,
                       JtZ, impt, sphereScale);
  if (!res) {
    return false;
  }

  cov = Jrotation * cameraRreference_covariance * Jrotation.transpose();
  cov += Jhfocal * hfocal_variance * Jhfocal.transpose();
  cov += Jvfocal * vfocal_variance * Jvfocal.transpose();
  cov += Jhcenter * hcenter_variance * Jhcenter.transpose();
  cov += Jvcenter * vcenter_variance * Jvcenter.transpose();
  cov += JdistortA * distorta_variance * JdistortA.transpose();
  cov += JdistortB * distortb_variance * JdistortB.transpose();
  cov += JdistortC * distortc_variance * JdistortC.transpose();
  cov += JtX * tx_variance * JtX.transpose();
  cov += JtY * ty_variance * JtY.transpose();
  cov += JtZ * tz_variance * JtZ.transpose();

  return true;
}

bool Camera::getProjectionCovariance(Eigen::Vector2d& mean, Eigen::Matrix2d& cov, const double hfocal_variance,
                                     const double vfocal_variance, const double hcenter_variance,
                                     const double vcenter_variance, const double distorta_variance,
                                     const double distortb_variance, const double distortc_variance,
                                     const double tx_variance, const double ty_variance, const double tz_variance,
                                     const Eigen::Vector3d& refpt, const Eigen::Matrix3d& refpt_cov) {
  bool res;

  Eigen::Matrix<double, 2, 3> Jpoint;
  Eigen::Matrix<double, 2, 1> Jhfocal;
  Eigen::Matrix<double, 2, 1> Jvfocal;
  Eigen::Matrix<double, 2, 1> Jhcenter;
  Eigen::Matrix<double, 2, 1> Jvcenter;
  Eigen::Matrix<double, 2, 1> JdistortA;
  Eigen::Matrix<double, 2, 1> JdistortB;
  Eigen::Matrix<double, 2, 1> JdistortC;
  Eigen::Matrix<double, 2, 9> Jrotation;
  Eigen::Matrix<double, 2, 1> JtX;
  Eigen::Matrix<double, 2, 1> JtY;
  Eigen::Matrix<double, 2, 1> JtZ;

  res = project_unbounded(mean, Jpoint, Jhfocal, Jvfocal, Jhcenter, Jvcenter, JdistortA, JdistortB, JdistortC,
                          Jrotation, JtX, JtY, JtZ, refpt);
  if (!res) {
    return false;
  }

  cov = Jrotation * cameraRreference_covariance * Jrotation.transpose();
  cov += Jhfocal * hfocal_variance * Jhfocal.transpose();
  cov += Jvfocal * vfocal_variance * Jvfocal.transpose();
  cov += Jhcenter * hcenter_variance * Jhcenter.transpose();
  cov += Jvcenter * vcenter_variance * Jvcenter.transpose();
  cov += JdistortA * distorta_variance * JdistortA.transpose();
  cov += JdistortB * distortb_variance * JdistortB.transpose();
  cov += JdistortC * distortc_variance * JdistortC.transpose();
  cov += Jpoint * refpt_cov * Jpoint.transpose();
  cov += JtX * tx_variance * JtX.transpose();
  cov += JtY * ty_variance * JtY.transpose();
  cov += JtZ * tz_variance * JtZ.transpose();

  return true;
}

}  // namespace Calibration
}  // namespace VideoStitch

#endif  // __clang_analyzer__
