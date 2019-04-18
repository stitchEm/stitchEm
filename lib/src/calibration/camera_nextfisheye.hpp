// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __CAMERA_NEXTFISHEYE__
#define __CAMERA_NEXTFISHEYE__

#include "camera.hpp"

namespace VideoStitch {
namespace Calibration {

/**
@brief Used to simulate a rig's camera.
*/
class Camera_NextFisheye : public Camera {
 protected:
  /**
  @brief Clone object
  */
  virtual Camera* clone() const;

  /**
  @brief Backproject from image plane to unit sphere space
  @param campt 3d point on the camera frame's unit sphere
  @param jacobian the transformation jacobian
  @param impt the input image coordinates (in meters)
  @return false if some numerical problem happened
  */
  virtual bool backproject(Eigen::Vector3d& campt, Eigen::Matrix<double, 3, 2>& jacobian, const Eigen::Vector2d& impt);

  /**
  @brief Project from sphere space to image plane in meters
  @param impt_meters projected point in meters
  @param jacobian the transformation jacobian
  @param campt the input 3d coordinates (in camera space)
  @return false if some numerical problem happened
  */
  virtual bool project(Eigen::Vector2d& impt_meters, Eigen::Matrix<double, 2, 3>& jacobian,
                       const Eigen::Vector3d& campt);
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
