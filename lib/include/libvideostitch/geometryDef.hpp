// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include "object.hpp"

namespace VideoStitch {
namespace Core {

class InputDefinition;

/**
 * Input Geometry definition
 */
class VS_EXPORT GeometryDefinition {
 public:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  GeometryDefinition();

  virtual ~GeometryDefinition();

  /**
   * @brief serialize
   * @param value A ptv value object to fill
   */
  void serialize(Ptv::Value& value) const;

  /**
   * Comparison operator.
   */
  bool operator==(const GeometryDefinition& other) const;
  bool hasSameExtrinsics(const GeometryDefinition& other) const;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param diff Input diff.
   * @param enforceMandatoryFields If false, ignore missing mandatory values.
   */
  Status applyDiff(const Ptv::Value& diff, bool enforceMandatoryFields);

  /**
   * @brief getDistortA
   * @return first radial distortion parameter
   */
  double getDistortA() const;

  /**
   * @brief setDistortA
   * @param distortA first radial distortion parameter
   */
  void setDistortA(const double distortA);

  /**
   * @brief getDistortB
   * @return second radial distortion parameter
   */
  double getDistortB() const;

  /**
   * @brief setDistortB
   * @param distortB second radial distortion parameter
   */
  void setDistortB(const double distortB);

  /**
   * @brief getDistortC
   * @return third radial distortion parameter
   */
  double getDistortC() const;

  /**
   * @brief setDistortC
   * @param distortC third radial distortion parameter
   */
  void setDistortC(const double distortC);

  /**
   * @brief getDistortP1
   * @return first tangential distortion parameter
   */
  double getDistortP1() const;

  /**
   * @brief setDistortP1
   * @param distortP1 first tangential distortion parameter
   */
  void setDistortP1(const double distortP1);

  /**
   * @brief getDistortP2
   * @return second tangential distortion parameter
   */
  double getDistortP2() const;

  /**
   * @brief setDistortP2
   * @param distortP2 second tangential distortion parameter
   */
  void setDistortP2(const double distortP2);

  /**
   * @brief getDistortS1
   * @return first thin-prism distortion parameter
   */
  double getDistortS1() const;

  /**
   * @brief setDistortS1
   * @param distortS1 first thin-prism distortion parameter
   */
  void setDistortS1(const double distortS1);

  /**
   * @brief getDistortS2
   * @return second thin-prism distortion parameter
   */
  double getDistortS2() const;

  /**
   * @brief setDistortS2
   * @param distortS2 second thin-prism distortion parameter
   */
  void setDistortS2(const double distortS2);

  /**
   * @brief getDistortS3
   * @return third thin-prism distortion parameter
   */
  double getDistortS3() const;

  /**
   * @brief setDistortS3
   * @param distortS3 third thin-prism distortion parameter
   */
  void setDistortS3(const double distortS3);

  /**
   * @brief getDistortS4
   * @return fourth thin-prism distortion parameter
   */
  double getDistortS4() const;

  /**
   * @brief setDistortS4
   * @param distortS4 fourth thin-prism distortion parameter
   */
  void setDistortS4(const double distortS4);

  /**
   * @brief getDistortTau1
   * @return first Scheimpflug distortion angle parameter in radians
   */
  double getDistortTau1() const;

  /**
   * @brief setDistortTau1
   * @param distortTau1 first Scheimpflug distortion angle parameter in radians
   */
  void setDistortTau1(const double distortTau1);

  /**
   * @brief getDistortTau2
   * @return second Scheimpflug distortion angle parameter in radians
   */
  double getDistortTau2() const;

  /**
   * @brief setDistortTau2
   * @param distortTau2 second Scheimpflug distortion angle parameter in radians
   */
  void setDistortTau2(const double distortTau2);

  /**
   * @brief hasNonRadialDistortion
   * @return true if any radial or non-radial distortion is not zero
   */
  bool hasDistortion() const;

  /**
   * @brief hasRadialDistortion
   * @return true if any radial distortion is not zero
   */
  bool hasRadialDistortion() const;

  /**
   * @brief hasNonRadialDistortion
   * @return true if any non-radial distortion is not zero
   */
  bool hasNonRadialDistortion() const;

  /**
   * @brief getHorizontalFocal
   * @return the horizontal focal in pixels
   */
  double getHorizontalFocal() const;

  /**
   * @brief setHorizontalFocal
   * @param focal horizontal focal in pixels to set
   */
  void setHorizontalFocal(const double focal);

  /**
   * @brief getEstimatedHorizontalFov
   * @param input input definition (used to get the lens type and image width)
   * @return the fov value estimated
   */
  double getEstimatedHorizontalFov(const InputDefinition& input) const;

  /**
   * @brief setEstimatedHorizontalFov set the focal according to the given fov
   * @param input input definition (used to get the lens type and image width)
   * @param fov new fov to set
   */
  void setEstimatedHorizontalFov(const InputDefinition& input, double fov);

  /**
   * @brief getVerticalFocal
   * @return the vertical focal in pixels
   */
  double getVerticalFocal() const;

  /**
   * @brief setVerticalFocal
   * @param focal vertical focal in pixels to set
   */
  void setVerticalFocal(const double focal);

  /**
   * @brief hasVerticalFov
   * @return true if vertical focal in pixels is defined
   */
  bool hasVerticalFocal() const;

  /**
   * @brief getCenterX
   * @return the center x coordinates
   */
  double getCenterX() const;

  /**
   * @brief setCenterX
   * @param centerx The x coordinate to update
   */
  void setCenterX(const double centerx);

  /**
   * @brief getCenterX
   * @return the center x coordinates
   */
  double getCenterY() const;

  /**
   * @brief setCenterY
   * @param centery The y coordinate to update
   */
  void setCenterY(const double centery);

  /**
   * @brief getYaw
   * @return The yaw rotation parameter in degrees
   */
  double getYaw() const;

  /**
   * @brief setYaw
   * @param yaw The yaw rotation parameter in degrees
   */
  void setYaw(const double yaw);

  /**
   * @brief getPitch
   * @return return Pitch rotation parameter in degrees
   */
  double getPitch() const;

  /**
   * @brief setPitch
   * @param pitch Pitch rotation parameter in degrees
   */
  void setPitch(const double pitch);

  /**
   * @brief getRoll
   * @return return Roll rotation parameter in degrees
   */
  double getRoll() const;

  /**
   * @brief setRoll
   * @param roll The Roll rotation parameter in degrees
   */
  void setRoll(const double roll);

  /**
   * @brief hasTranslation
   * @return true if translation is defined
   */
  bool hasTranslation() const;

  /**
   * @brief getTranslationX
   * @return return X component of the camera translation
   */
  double getTranslationX() const;

  /**
   * @brief setRoll
   * @param X X component of the camera translation
   */
  void setTranslationX(const double X);

  /**
   * @brief getTranslationX
   * @return return Y component of the camera translation
   */
  double getTranslationY() const;

  /**
   * @brief setRoll
   * @param Y Y component of the camera translation
   */
  void setTranslationY(const double Y);

  /**
   * @brief getTranslationZ
   * @return return Z component of the camera translation
   */
  double getTranslationZ() const;

  /**
   * @brief setRoll
   * @param Z Z component of the camera translation
   */
  void setTranslationZ(const double Z);

  /**
   * Gets a geometric value using its id
   * Valid ids are 'y' (yaw), 'p' (pitch), 'r' (roll), 'f' (hfov), 'a' (lensDistA), 'b' (lensDistB), 'c' (lensDistC),
   * 'd' (distCenterX), 'e' (distCenterY) Returns 0.0 if the id is not valid
   * @param id of the parameter to retrieve
   */
  double getGeoParamFromId(const char id) const;

  /**
   * Sets a geometric value using its id
   * Valid ids are 'y' (yaw), 'p' (pitch), 'r' (roll), 'f' (hfov), 'a' (lensDistA), 'b' (lensDistB), 'c' (lensDistC),
   * 'd' (distCenterX), 'e' (distCenterY) No operation is performed if id is not valid
   * @param id
   * @param val A value to set for the given parameter
   */
  void setGeoParamFromId(const char id, const double val);

  /**
   * Applies a global orientation to Yaw/Pitch/Roll angles
   * @param orientation quaternion of the global orientation
   */
  void applyGlobalOrientation(const Quaternion<double>& orientation);

  /**
   * Reset to 0 all parameters except the focal parameters
   */
  void resetAllButFocal();

  /**
   * Reset to 0 all extrinsic parameters
   */
  void resetExtrinsics();

  /**
   * Resets all distortion parameters and the center to 0
   */
  void resetDistortion();

  /**
   * Convert loaded fov if it exists to focal length in pixels
   * @param input definition (to know the lens type)
   */
  void convertLoadedFovToFocal(const InputDefinition& input);

 private:
  double horizontalFocal;

  bool doesHaveVerticalFocal;
  double verticalFocal;

  double center_x;
  double center_y;

  double distort_a;
  double distort_b;
  double distort_c;

  double distort_p1;
  double distort_p2;
  double distort_s1;
  double distort_s2;
  double distort_s3;
  double distort_s4;
  double distort_tau1;
  double distort_tau2;

  double yaw;
  double pitch;
  double roll;

  bool doesHaveTranslation;
  double translation_x;
  double translation_y;
  double translation_z;

  bool hasFovLoaded;
};

}  // namespace Core
}  // namespace VideoStitch
