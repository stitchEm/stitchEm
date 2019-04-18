// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <array>
#include <memory>
#include <Eigen/Dense>

#include "libvideostitch/config.hpp"
#include "libvideostitch/inputDef.hpp"

#include "boundedValue.hpp"

namespace VideoStitch {

namespace Core {
class GeometryDefinition;
class RigCameraDefinition;
}  // namespace Core

namespace Calibration {

/**
Used to simulate a rig's camera.
*/
class VS_EXPORT Camera {
 public:
  Camera();
  virtual ~Camera();
  virtual Camera* clone() const = 0;

  /**
  @brief Setup camera instance using parameters from a camera rigcamera definition.
  @note These parameters will be transformed to optimize computations
  @param rigcamdef input camera definition
  */
  void setupWithRigCameraDefinition(Core::RigCameraDefinition& rigcamdef);

  /**
  @brief Effectively lift a point on the sphere of radius sphereScale in the reference frame
  @param refpt the result lifted point
  @param Jhfocal jacobian of the function wrt horizontal focal
  @param Jvfocal jacobian of the function wrt vertical focal
  @param Jhcenter jacobian of the function wrt horizontal center
  @param Jvcenter jacobian of the function wrt vertical center
  @param JdistortA jacobian of the function wrt distortion param A
  @param JdistortB jacobian of the function wrt distortion param B
  @param JdistortC jacobian of the function wrt distortion param C
  @param Jrotation jacobian of the function wrt rotation params
  @param JtX jacobian of the function wrt translation X param
  @param JtY jacobian of the function wrt translation Y param
  @param JtZ jacobian of the function wrt translation Z param
  @param impt the input image point
  @param sphereScale the sphere radius in the reference frame
  @return false if some numerical problem happened
  */
  bool lift(Eigen::Vector3d& refpt, Eigen::Matrix<double, 3, 4>& Jhfocal, Eigen::Matrix<double, 3, 4>& Jvfocal,
            Eigen::Matrix<double, 3, 4>& Jhcenter, Eigen::Matrix<double, 3, 4>& Jvcenter,
            Eigen::Matrix<double, 3, 4>& JdistortA, Eigen::Matrix<double, 3, 4>& JdistortB,
            Eigen::Matrix<double, 3, 4>& JdistortC, Eigen::Matrix<double, 3, 9>& Jrotation,
            Eigen::Matrix<double, 3, 4>& JtX, Eigen::Matrix<double, 3, 4>& JtY, Eigen::Matrix<double, 3, 4>& JtZ,
            const Eigen::Vector2d& impt, const double sphereScale);

  /**
   @brief Effectively lift a point on the sphere of radius sphereScale in the reference frame
   @param refpt the result lifted point
   @param impt the input image point
   @param sphereScale the sphere radius in the reference frame
   @return false if some numerical problem happened
   */
  bool lift(Eigen::Vector3d& refpt, const Eigen::Vector2d& impt, const double sphereScale);

  /**
  @brief Effectively lift a point on the unit sphere in the camera frame
  @param campt the result lifted point
  @param impt the input image point
  @return false if some numerical problem happened
  */
  bool quicklift(Eigen::Vector3d& campt, const Eigen::Vector2d& impt);

  /**
  @brief Project a point on the unit sphere in the reference space into the camera plane
  @param impt_pixels the result projected point
  @param Jpoint jacobian of the function wrt source point
  @param Jhfocal jacobian of the function wrt horizontal focal
  @param Jvfocal jacobian of the function wrt vertical focal
  @param Jhcenter jacobian of the function wrt horizontal center
  @param Jvcenter jacobian of the function wrt vertical center
  @param JdistortA jacobian of the function wrt lens distortion param A
  @param JdistortB jacobian of the function wrt lens distortion param B
  @param JdistortC jacobian of the function wrt lens distortion param C
  @param Jrotation jacobian of the function wrt rotation params
  @param JtX jacobian of the function wrt translation X param
  @param JtY jacobian of the function wrt translation Y param
  @param JtZ jacobian of the function wrt translation Z param
  @param refpt the input point in the reference space
  @return false if some numerical problem happened
  */
  bool project(Eigen::Vector2d& impt_pixels, Eigen::Matrix<double, 2, 3>& Jpoint, Eigen::Matrix<double, 2, 4>& Jhfocal,
               Eigen::Matrix<double, 2, 4>& Jvfocal, Eigen::Matrix<double, 2, 4>& Jhcenter,
               Eigen::Matrix<double, 2, 4>& Jvcenter, Eigen::Matrix<double, 2, 4>& JdistortA,
               Eigen::Matrix<double, 2, 4>& JdistortB, Eigen::Matrix<double, 2, 4>& JdistortC,
               Eigen::Matrix<double, 2, 9>& Jrotation, Eigen::Matrix<double, 2, 4>& JtX,
               Eigen::Matrix<double, 2, 4>& JtY, Eigen::Matrix<double, 2, 4>& JtZ, const Eigen::Vector3d& refpt);

  /**
   @brief Project a point on the unit sphere in the reference space into the camera plane
   @param impt_pixels the result projected point
   @param refpt the input point in the reference space
   @return false if some numerical problem happened
   */
  bool project(Eigen::Vector2d& impt_pixels, const Eigen::Vector3d& refpt);

  /**
   @brief Project a point on the unit sphere in the camera space into the camera plane
   @param impt_pixels the result projected point
   @param refpt the input point in the reference space
   @return false if some numerical problem happened
   */
  bool quickproject(Eigen::Vector2d& impt_pixels, const Eigen::Vector3d& campt);

  /**
  @brief Get point lift mean and covariance matrix
  @param mean the lifted point
  @param hfocal_variance the variance of horizontal focal parameter
  @param vfocal_variance the variance of vertical focal parameter
  @param hcenter_variance the variance of horizontal center parameter
  @param vcenter_variance the variance of vertical center parameter
  @param distorta_variance the variance of distortion A parameter
  @param distortb_variance the variance of distortion B parameter
  @param distortc_variance the variance of distortion C parameter
  @param tx_variance the variance of translation X parameter
  @param ty_variance the variance of translation Y parameter
  @param tz_variance the variance of translation Z parameter
  @param impt the original 2d point to lift
  @param sphereScale the sphere radius in the reference frame
  @return false if an error occured
  */
  bool getLiftCovariance(Eigen::Vector3d& mean, Eigen::Matrix3d& cov, const double hfocal_variance,
                         const double vfocal_variance, const double hcenter_variance, const double vcenter_variance,
                         const double distorta_variance, const double distortb_variance, const double distortc_variance,
                         const double tx_variance, const double ty_variance, const double tz_variance,
                         const Eigen::Vector2d& impt, const double sphereScale);

  /**
  @brief Get point projection mean and covariance matrix
  @param mean the lifted point
  @param hfocal_variance the variance of horizontal focal parameter
  @param vfocal_variance the variance of vertical focal parameter
  @param hcenter_variance the variance of horizontal center parameter
  @param vcenter_variance the variance of vertical center parameter
  @param distorta_variance the variance of distortion A parameter
  @param distortb_variance the variance of distortion B parameter
  @param distortc_variance the variance of distortion C parameter
  @param tx_variance the variance of translation X parameter
  @param ty_variance the variance of translation Y parameter
  @param tz_variance the variance of translation Z parameter
  @param refpt the original 3d point to project
  @param refpt_cov the original 2d point covariance to project
  @return false if an error occured
  */
  bool getProjectionCovariance(Eigen::Vector2d& mean, Eigen::Matrix2d& cov, const double hfocal_variance,
                               const double vfocal_variance, const double hcenter_variance,
                               const double vcenter_variance, const double distorta_variance,
                               const double distortb_variance, const double distortc_variance, const double tx_variance,
                               const double ty_variance, const double tz_variance, const Eigen::Vector3d& refpt,
                               const Eigen::Matrix3d& refpt_cov);

  /**
  @brief Returns whether the horizontal focal is constant
  */
  bool isHorizontalFocalConstant() const;

  /**
  @brief Get a pointer on horizontal focal parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getHorizontalFocalPtr();

  /**
  @brief Set lens horizontal focal vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setHorizontalFocal(const double* ptr);

  /**
  @brief Returns whether the vertical focal is constant
  */
  bool isVerticalFocalConstant() const;

  /**
  @brief Get a pointer on horizontal focal parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getVerticalFocalPtr();

  /**
  @brief Set lens vertical focal vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setVerticalFocal(const double* ptr);

  /**
  @brief Returns whether the horizontal center is constant
  */
  bool isHorizontalCenterConstant() const;

  /**
  @brief Get a pointer on horizontal center parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getHorizontalCenterPtr();

  /**
  @brief Set lens horizontal center vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setHorizontalCenter(const double* ptr);

  /**
  @brief Returns whether the vertical center is constant
  */
  bool isVerticalCenterConstant() const;

  /**
  @brief Get a pointer on horizontal center parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getVerticalCenterPtr();

  /**
  @brief Set lens vertical center vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setVerticalCenter(const double* ptr);

  /**
  @brief Returns whether the First distortion parameter is constant
  */
  bool isDistortionAConstant() const;

  /**
  @brief Get a pointer on First distortion parameter parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getDistortionAPtr();

  /**
  @brief Set First distortion parameter vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setDistortionA(const double* ptr);

  /**
  @brief Returns whether the Second distortion parameter is constant
  */
  bool isDistortionBConstant() const;

  /**
  @brief Get a pointer on Second distortion parameter parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getDistortionBPtr();

  /**
  @brief Set Second distortion parameter vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setDistortionB(const double* ptr);

  /**
  @brief Returns whether the Third distortion parameter is constant
  */
  bool isDistortionCConstant() const;

  /**
  @brief Get a pointer on Third distortion parameter parameters vector
  @return a pointer to double vector (4 elements)
  */
  double* getDistortionCPtr();

  /**
  @brief Set Third distortion parameter vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setDistortionC(const double* ptr);

  /**
  @brief Returns whether the camera rotation is constant
  */
  bool isRotationConstant() const;

  /**
  @brief Get a pointer on camera rotation vector
  @return a pointer to double vector (9 elements)
  */
  double* getRotationPtr();

  /**
  @brief Set rotation vector from input vector pointer
  @param ptr pointer to the input data
  */
  void setRotation(const double* ptr);

  /**
  @brief Get rotation matrix
  @return 3x3 matrix
  */
  Eigen::Matrix3d getRotation() const;

  /**
  @brief Set rotation matrix directly
  @param R the input matrix to copy
  */
  void setRotationMatrix(const Eigen::Matrix3d& R);

  /**
  @brief Set rotation matrix from the presets
  */
  void setRotationFromPresets();

  /**
   @brief Checks if given rotation is within presets
   @param R the input matrix to check
   @return boolean if rotation is within presets
   */
  bool isRotationWithinPresets(const Eigen::Matrix3d& R) const;

  /**
   @brief Get reference pose matrix from presets
   @return 3x3 matrix
   */
  Eigen::Matrix3d getRotationFromPresets() const;

  /**
   @brief Returns whether the camera translation X is constant
   */
  bool isTranslationXConstant() const;

  /**
   @brief Get a pointer camera translation X component
   @return a pointer to double
   */
  double* getTranslationXPtr();

  /**
   @brief Set camera translation X from input vector pointer
   @param ptr pointer to the input data
   */
  void setTranslationX(const double* ptr);

  /**
   @brief Returns whether the camera translation Y is constant
   */
  bool isTranslationYConstant() const;

  /**
   @brief Get a pointer camera translation Y component
   @return a pointer to double
   */
  double* getTranslationYPtr();

  /**
   @brief Set camera translation Y from input vector pointer
   @param ptr pointer to the input data
   */
  void setTranslationY(const double* ptr);

  /**
   @brief Returns whether the camera translation Z is constant
   */
  bool isTranslationZConstant() const;

  /**
   @brief Get a pointer camera translation Z component
   @return a pointer to double
   */
  double* getTranslationZPtr();

  /**
   @brief Set camera translation Z from input vector pointer
   @param ptr pointer to the input data
   */
  void setTranslationZ(const double* ptr);

  /**
   @brief Get camera translation in a single output vector
   @return camera translation vector
   */
  Eigen::Vector3d getTranslation() const;

  /**
   @brief Set camera translations from input vector
   @param translation input translation vector
   */
  void setTranslation(const Eigen::Vector3d& translation);

  /**
  @brief Fill Geometry
  @param geometry the to update geometry
  @param width width of the input image
  @param height height of the input image
  */
  void fillGeometry(Core::GeometryDefinition& geometry, int width, int height);

  /**
  @brief Get Image width
  @return width of captured image
  */
  size_t getWidth();

  /**
  @brief Get Image height
  @return height of captured image
  */
  size_t getHeight();

  /**
   @brief Get the rotation covariance matrix
   @param cov the estimated covariance
   */
  void getRotationCovarianceMatrix(Eigen::Matrix<double, 9, 9>& cov) const;

  /**
   @brief Get the yaw/pitch/roll covariance matrix
   @param cov the covariance
   */
  void getYawPitchRollCovarianceMatrix(Eigen::Matrix<double, 3, 3>& cov) const;

  /**
  @brief Get relative rotation
  @param second_Rmean_first relative rotation mean between 2 cameras
  @param second_axisAngleRCov_first relative rotation variance between 2 cameras in axis-angle representation
  @param first the first camera
  @param second the second camera
  */
  static void getRelativeRotation(Eigen::Matrix3d& second_Rmean_first, Eigen::Matrix3d& second_axisAngleRCov_first,
                                  const Camera& first, const Camera& second);

  /*
  @brief Get Format for this camera
  @return format for this camera class
  */
  Core::InputDefinition::Format getFormat();

  /**
  @brief Set Format for this camera
  @param format this camera format
  */
  void setFormat(Core::InputDefinition::Format format);

  /**
  @brief Ties the focal to the focal of another camera
  @param other camera to have the focal tied to
  */
  void tieFocalTo(const Camera& other);

  /**
  @brief Untie the focal values (i.e. will have its own values independent from other objects)
  */
  void untieFocal();

 protected:
  /**
  @brief Effectively lift a point on the sphere of radius sphereScale in the reference frame
  @param refpt the result lifted point
  @param Jhfocal jacobian of the function wrt horizontal focal
  @param Jvfocal jacobian of the function wrt vertical focal
  @param Jhcenter jacobian of the function wrt horizontal center
  @param Jvcenter jacobian of the function wrt vertical center
  @param JdistortA jacobian of the function wrt distortion param A
  @param JdistortB jacobian of the function wrt distortion param B
  @param JdistortC jacobian of the function wrt distortion param C
  @param Jrotation jacobian of the function wrt lens distortion params
  @param Jpose jacobian of the function wrt pose params
  @param impt the input image point
  @param sphereScale the sphere radius in the reference frame
  @return false if some numerical problem happened
  */
  bool lift_unbounded(Eigen::Vector3d& refpt, Eigen::Matrix<double, 3, 1>& Jhfocal,
                      Eigen::Matrix<double, 3, 1>& Jvfocal, Eigen::Matrix<double, 3, 1>& Jhcenter,
                      Eigen::Matrix<double, 3, 1>& Jvcenter, Eigen::Matrix<double, 3, 1>& JdistortA,
                      Eigen::Matrix<double, 3, 1>& JdistortB, Eigen::Matrix<double, 3, 1>& JdistortC,
                      Eigen::Matrix<double, 3, 9>& Jrotation, Eigen::Matrix<double, 3, 1>& JtranslationX,
                      Eigen::Matrix<double, 3, 1>& JtranslationY, Eigen::Matrix<double, 3, 1>& JtranslationZ,
                      const Eigen::Vector2d& impt, const double sphereScale);

  /**
  @brief Project a point on the unit sphere in the reference space into the camera plane
  @param impt the result projected point
  @param Jpoint jacobian of the function wrt source point
  @param Jhfocal jacobian of the function wrt horizontal focal
  @param Jvfocal jacobian of the function wrt vertical focal
  @param Jhcenter jacobian of the function wrt horizontal center
  @param Jvcenter jacobian of the function wrt vertical center
  @param JdistortA jacobian of the function wrt lens distortion param A
  @param JdistortB jacobian of the function wrt lens distortion param B
  @param JdistortC jacobian of the function wrt lens distortion param C
  @param Jrotation jacobian of the function wrt pose params
  @param refpt the input point in the reference space
  @return false if some numerical problem happened
  */
  bool project_unbounded(Eigen::Vector2d& impt_pixels, Eigen::Matrix<double, 2, 3>& Jpoint,
                         Eigen::Matrix<double, 2, 1>& Jhfocal, Eigen::Matrix<double, 2, 1>& Jvfocal,
                         Eigen::Matrix<double, 2, 1>& Jhcenter, Eigen::Matrix<double, 2, 1>& Jvcenter,
                         Eigen::Matrix<double, 2, 1>& JdistortA, Eigen::Matrix<double, 2, 1>& JdistortB,
                         Eigen::Matrix<double, 2, 1>& JdistortC, Eigen::Matrix<double, 2, 9>& Jrotation,
                         Eigen::Matrix<double, 2, 1>& JtranslationX, Eigen::Matrix<double, 2, 1>& JtranslationY,
                         Eigen::Matrix<double, 2, 1>& JtranslationZ, const Eigen::Vector3d& refpt);

  /**
  @brief Backproject from image plane to unit sphere space
  @param campt 3d point on the camera frame's unit sphere
  @param jacobian the transformation jacobian
  @param impt the input image coordinates (in meters)
  @return false if some numerical problem happened
  */
  virtual bool backproject(Eigen::Vector3d& campt, Eigen::Matrix<double, 3, 2>& jacobian,
                           const Eigen::Vector2d& impt) = 0;

  /**
  @brief Project from unit sphere space to image plane in meters
  @param impt_meters projected point in meters
  @param jacobian the transformation jacobian
  @param campt the input 3d coordinates (in camera space)
  @return false if some numerical problem happened
  */
  virtual bool project(Eigen::Vector2d& impt_meters, Eigen::Matrix<double, 2, 3>& jacobian,
                       const Eigen::Vector3d& campt) = 0;

  /**
  @brief Undistort a point
  @param impt_undistorted output undistorted
  @param impt_distorted input image coordinates
  */
  bool undistort(Eigen::Vector2d& impt_undistorted, const Eigen::Vector2d& impt_distorted);

  /**
  @brief Undistort a point
  @param impt_undistorted output undistorted
  @param Jundistortwrtdistorted jacobian of undistorted wrt distorted coordinates
  @param JparameterA jacobian of undistorted wrt A
  @param JparameterB jacobian of undistorted wrt B
  @param JparameterC jacobian of undistorted wrt C
  @param impt_distorted input image coordinates
  */
  bool undistort(Eigen::Vector2d& impt_undistorted, Eigen::Matrix<double, 2, 2>& Jundistortwrtdistorted,
                 Eigen::Matrix<double, 2, 1>& JparameterA, Eigen::Matrix<double, 2, 1>& JparameterB,
                 Eigen::Matrix<double, 2, 1>& JparameterC, const Eigen::Vector2d& impt_distorted);

  /**
  @brief Undistort a point
  @param impt_distorted output undistorted
  @param impt_undistorted input image coordinates
  */
  bool distort(Eigen::Vector2d& impt_distorted, const Eigen::Vector2d& impt_undistorted);

  /**
  @brief Undistort a point
  @param impt_distorted output undistorted
  @param Jdistortedwrtundistorted jacobian of undistorted wrt distorted coordinates
  @param JparameterA jacobian of undistorted wrt A
  @param JparameterB jacobian of undistorted wrt B
  @param JparameterC jacobian of undistorted wrt C
  @param impt_undistorted input image coordinates
  */
  bool distort(Eigen::Vector2d& impt_distorted, Eigen::Matrix<double, 2, 2>& Jdistortedwrtundistorted,
               Eigen::Matrix<double, 2, 1>& JparameterA, Eigen::Matrix<double, 2, 1>& JparameterB,
               Eigen::Matrix<double, 2, 1>& JparameterC, const Eigen::Vector2d& impt_undistorted);

 protected:
  /*
  WARNING: if you add a member here, do not forget to update the clone() method in all derived classes
  */
  BoundedValue horizontal_focal;
  BoundedValue vertical_focal;
  BoundedValue horizontal_center;
  BoundedValue vertical_center;
  BoundedValue distort_A;
  BoundedValue distort_B;
  BoundedValue distort_C;
  BoundedValue t_x;
  BoundedValue t_y;
  BoundedValue t_z;

  Eigen::Vector3d yprReference;
  Eigen::Matrix<double, 3, 3> yprCovariance;
  Eigen::Matrix<double, 3, 3> cameraRreference;
  Eigen::Matrix<double, 9, 9> cameraRreference_covariance;
  std::array<double, 9> cameraR_data;
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > cameraR;

  Core::InputDefinition::Format format;
  size_t width;
  size_t height;
};

}  // namespace Calibration
}  // namespace VideoStitch
