// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "object.hpp"
#include "cameraDef.hpp"

#include <map>
#include <memory>

namespace VideoStitch {
namespace Ptv {
class Value;
}
namespace Core {

class GeometryDefinition;

/**
 * @brief A Rig camera model representation class.
 */
class VS_EXPORT RigCameraDefinition : public Ptv::Object {
 public:
  RigCameraDefinition();
  ~RigCameraDefinition();

  /**
   * Copy constructor
   * @param other the original def
   */
  RigCameraDefinition(const RigCameraDefinition& other);

  /**
   * Assignment operator
   */
  RigCameraDefinition& operator=(const RigCameraDefinition&);

  /**
   * Build from a Ptv::Value.
   * @param camerasmap A map with cameras definition indexed by their names
   * @param value Input value.
   * @return The parsed rig, or NULL on error.
   */
  static RigCameraDefinition* create(const std::map<std::string, std::shared_ptr<CameraDefinition> >& camerasmap,
                                     const Ptv::Value& value);

  /**
   * Serialize the rig camera definition into a PTV::Value
   * @return a ptv value
   */
  Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  bool operator==(const RigCameraDefinition& other) const;

  /**
   * Validate that the rig makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  bool validate(std::ostream& os) const;

  /**
   * Get Yaw value of the camera instance in radians
   * @return yaw value
   */
  NormalDouble getYawRadians() const;

  /**
   * Set Yaw value for the camera instance in radians
   * @param value the update uncertain value
   */
  void setYawRadians(const NormalDouble& value);

  /**
   * Get Pitch value of the camera instance
   * @return Pitch value
   */
  NormalDouble getPitchRadians() const;

  /**
   * Set Pitch value for the camera instance in radians
   * @param value the update uncertain value
   */
  void setPitchRadians(const NormalDouble& value);

  /**
   * Get Roll value of the camera instance in radians
   * @return roll value
   */
  NormalDouble getRollRadians() const;

  /**
   * Set Roll value for the camera instance in radians
   * @param value the update uncertain value
   */
  void setRollRadians(const NormalDouble& value);

  /**
   * Get translation X value of the camera instance
   * @return translation X value
   */
  NormalDouble getTranslationX() const;

  /**
   * Set translation X value for the camera instance
   * @param value the update uncertain value
   */
  void setTranslationX(const NormalDouble& value);

  /**
   * Get translation Y value of the camera instance
   * @return translation Y value
   */
  NormalDouble getTranslationY() const;

  /**
   * Set translation Y value for the camera instance
   * @param value the update uncertain value
   */
  void setTranslationY(const NormalDouble& value);

  /**
   * Get translation Z value of the camera instance
   * @return translation Z value
   */
  NormalDouble getTranslationZ() const;

  /**
   * Set translation Z value for the camera instance
   * @param value the update uncertain value
   */
  void setTranslationZ(const NormalDouble& value);

  /**
   * Set camera instance
   * @param value the new camera instance
   */
  void setCamera(const std::shared_ptr<CameraDefinition>& value);

  /**
   * Get Camera instance
   * @return instance
   */
  std::shared_ptr<CameraDefinition> getCamera() const;

  /**
   * Deserialize a rig camera definition
   * @param camerasmap the set of camera presets
   * @param value the input ptv
   * @return  true if succeeded
   */
  bool deserialize(const std::map<std::string, std::shared_ptr<CameraDefinition> >& camerasmap,
                   const Ptv::Value& value);

  /**
   * Using the mean value, fill the geometry definition
   * @param def the geometry definition to fill
   * @return false if an error occured
   */
  bool fillGeometryDefinition(GeometryDefinition& def) const;

 public:
  /**
  Constant value for max bound value
  */
  static const double max_yaw_variance;

  /**
  Constant value for max bound value
  */
  static const double max_pitch_variance;

  /**
  Constant value for max bound value
  */
  static const double max_roll_variance;

  /**
   Constant value for max bound value
   */
  static const double max_translation_x_variance;

  /**
   Constant value for max bound value
   */
  static const double max_translation_y_variance;

  /**
   Constant value for max bound value
   */
  static const double max_translation_z_variance;

 private:
  class Pimpl;
  Pimpl* const pimpl;
};
}  // namespace Core
}  // namespace VideoStitch
