// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "object.hpp"
#include <vector>
#include "cameraDef.hpp"
#include "panoDef.hpp"
#include <memory>

namespace VideoStitch {
namespace Ptv {
class Value;
}
namespace Core {

class RigCameraDefinition;

/**
 * @brief A Rig model representation class.
 */
class VS_EXPORT RigDefinition : public Ptv::Object {
 public:
  RigDefinition();
  ~RigDefinition();

  /**
   * @brief Clones a rig java-style.
   * @return A similar rig. Ownership is given to the caller.
   */
  RigDefinition* clone() const;

  /**
   * @brief Build from a Ptv::Value.
   * @param camerasmap A map with cameras definition indexed by their names
   * @param value Input value.
   * @return The parsed rig, or NULL on error.
   */
  static RigDefinition* create(const std::map<std::string, std::shared_ptr<CameraDefinition>>& camerasmap,
                               const Ptv::Value& value);

  /**
   * @brief Build a rig with basic params
   * @param name the name of the rig
   * @param lensformat the cameras lens format
   * @param nbcameras number of cameras in this
   * @param image_width width of the acquired image
   * @param image_height height of the acquired image
   * @param cropped_width width of the cropped image
   * @param cropped_height height of the cropped image
   * @param fov base horizontal fov in degrees (same init for all cameras)
   * @param pano an optional panorama definition, to initialize the pose of the cameras
   * @return The newly created rig, or NULL on error.
   */
  static RigDefinition* createBasicUnknownRig(const std::string& name, const InputDefinition::Format& lensformat,
                                              size_t nbcameras, size_t image_width, size_t image_height,
                                              size_t cropped_width, size_t cropped_height, double fov,
                                              const PanoDefinition* pano = nullptr);

  /**
   * @brief Build a rig from a PanoDefinition considered as a template
   * @param name the name of the rig
   * @param focalStdDevValuePercentage standard deviation of focal expressed in percentage of focal value
   * @param centerStdDevWidthPercentage standard deviation of center expressed in percentage of input width
   * @param distortStdDevValuePercentage standard deviation of distortion expressed in percentage of distortion value
   * @param yawStdDev standard deviation of yaw angle in degrees
   * @param pitchStdDev standard deviation of pitch angle in degrees
   * @param rollStdDev standard deviation of roll angle in degrees
   * @param translationXStdDev standard deviation of translation X
   * @param translationYStdDev standard deviation of translation Y
   * @param translationZStdDev standard deviation of translation Z
   * @param pano a panorama definition used as a template
   * @param applyGlobalOrientation boolean to apply or not the PanoDefinition global orientation
   * @return The newly created rig, or NULL on error.
   */
  static RigDefinition* createFromPanoDefinitionTemplate(
      const std::string& name, const double focalStdDevValuePercentage, const double centerStdDevWidthPercentage,
      const double distortStdDevValuePercentage, const double yawStdDevDegrees, const double pitchStdDevDegrees,
      const double rollStdDevDegrees, const double translationXStdDev, const double translationYStdDev,
      const double translationZStdDev, const PanoDefinition& pano, const bool applyPanoGlobalOrientation);

  /**
   * @brief Override the standard deviations of a rig
   * @param focalStdDevValuePercentage standard deviation of focal expressed in percentage of focal value
   * @param centerStdDevWidthPercentage standard deviation of center expressed in percentage of input width
   * @param distortStdDevValuePercentage standard deviation of distortion expressed in percentage of distortion value
   * @param yawStdDev standard deviation of yaw angle in degrees
   * @param pitchStdDev standard deviation of pitch angle in degrees
   * @param rollStdDev standard deviation of roll angle in degrees
   * @param translationXStdDev standard deviation of translation X
   * @param translationYStdDev standard deviation of translation Y
   * @param translationZStdDev standard deviation of translation Z
   */
  void overridePresetsStandardDeviations(const double focalStdDevValuePercentage,
                                         const double centerStdDevWidthPercentage,
                                         const double distortStdDevValuePercentage, const double yawStdDevDegrees,
                                         const double pitchStdDevDegrees, const double rollStdDevDegrees,
                                         const double translationXStdDev, const double translationYStdDev,
                                         const double translationZStdDev);

  /**
   * @brief Serialize the rig definition into a PTV::Value
   * @return a ptv value
   */
  Ptv::Value* serialize() const;

  /**
   * @brief Comparison operator.
   */
  bool operator==(const RigDefinition& other) const;

  /**
   * @brief Validate that the rig makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  bool validate(std::ostream& os, const size_t numCameras) const;

  /**
   * @brief Get the rig definition name
   * @return a name as string
   */
  std::string getName() const;

  /**
   * @brief Set the rig definition name
   * @param name the new value to set
   */
  void setName(const std::string& name);

  /**
   * @brief Get a reference to the nth rig camera definition
   * @param cam the result definition
   * @param n the index of the definition
   * @return false if this index is out of bounds
   */
  bool getRigCameraDefinition(RigCameraDefinition& cam, size_t n) const;

  /**
   * @brief Get a map of camera definitions indexed by name
   * @return a map of camera definitions indexed by name
   */
  std::map<std::string, std::shared_ptr<CameraDefinition>> getRigCameraDefinitionMap() const;

  /**
  @brief How many camera are defined for this rig ?
  @return the number of cameras
  */
  size_t getRigCameraDefinitionCount() const;

  /**
   * @brief Clears the camera definition list
   */
  void removeRigCameraDefinition();

 protected:
  /**
   * Disabled, use clone()
   */
  RigDefinition(const RigDefinition&) = delete;
  /**
   * Disabled, use clone()
   */
  RigDefinition& operator=(const RigDefinition&) = delete;

 private:
  class Pimpl;
  Pimpl* const pimpl;
};
}  // namespace Core
}  // namespace VideoStitch
