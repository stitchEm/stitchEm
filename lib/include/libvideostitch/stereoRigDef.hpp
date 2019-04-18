// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "object.hpp"

#include <vector>

namespace VideoStitch {
namespace Ptv {
class Value;
}
namespace Core {

/**
 * @brief A stereo rig representation class.
 */
class VS_EXPORT StereoRigDefinition : public Ptv::Object {
 public:
  /**
   * Orientation of the camera in the horizontal plane.
   */
  enum Orientation { Portrait, Landscape, Portrait_flipped, Landscape_flipped };

  /**
   * Geometry of the rig.
   */
  enum Geometry { Circular, Polygonal };

  ~StereoRigDefinition();

  /**
   * Clones a Rig java-style.
   * @return A similar Rig. Ownership is given to the caller.
   */
  StereoRigDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed Rig, or NULL on error.
   */
  static StereoRigDefinition* create(const Ptv::Value& value);

  Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  bool operator==(const StereoRigDefinition& other) const;

  /**
   * Validate that the rig makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  bool validate(std::ostream& os) const;

  /**
   * Returns the orientation of the cameras in the horizontal plane.
   */
  Orientation getOrientation() const;

  /**
   * Returns the geometry of the rig.
   */
  Geometry getGeometry() const;

  /**
   * Returns the diameter of the rig.
   * @note Only used for circular rigs.
   */
  double getDiameter() const;

  /**
   * Returns the inter-pupillary distance expected.
   */
  double getIPD() const;

  /**
   * Returns the set of inputs for the left eye.
   */
  std::vector<int> getLeftInputs() const;

  /**
   * Returns the set of inputs for the right eye.
   */
  std::vector<int> getRightInputs() const;

  /**
   * Sets the orientation of the cameras in the horizontal plane.
   */
  void setOrientation(Orientation);

  /**
   * Sets the geometry of the rig.
   */
  void setGeometry(Geometry);

  /**
   * Sets the diameter of the rig.
   * @note Only used for circular rigs.
   */
  void setDiameter(double);

  /**
   * Sets the expected inter-pupillary distance.
   */
  void setIPD(double);

  /**
   * Sets the left inputs list
   */
  void setLeftInputs(const std::vector<int>& left);

  /**
   * Sets the right inputs list
   */
  void setRightInputs(const std::vector<int>& right);

  /**
   * Auxiliary methods for converting enum values into strings and vice-versa
   */

  /**
   * Set geometry enum by providing its name as a string
   * @param name name of the geomtry
   * @param geom geometry to be set if a valid name is provided
   * @return Returns true on success, false if the given name is not known or invalid
   */
  static bool getGeometryFromName(const std::string& name, Geometry& geom);

  /**
   * Set orientatn enum by providing its name as a string
   * @param name name of the orientation
   * @param orient orientation to be set if a valid name is provided
   * @return Returns true on success, false if the given name is not known or invalid
   */
  static bool getOrientationFromName(const std::string& name, Orientation& orient);

  /**
   * Get orientation name from its enum value
   * @param orient Orientation enum to be converted to string
   * @return Returns name for the given orientation enum
   */
  static const std::string getOrientationName(const Orientation orient);

  /**
   * Get geometry name from its enum value
   * @param geom Geometry enum to be converted to string
   * @return Returns name for the given Geometry enum
   */
  static const std::string getGeometryName(const Geometry geom);

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  StereoRigDefinition();

  /**
   * Disabled, use clone()
   */
  StereoRigDefinition(const StereoRigDefinition&) = delete;
  /**
   * Disabled, use clone()
   */
  StereoRigDefinition& operator=(const StereoRigDefinition&) = delete;

 private:
  class Pimpl;
  Pimpl* const pimpl;
};
}  // namespace Core
}  // namespace VideoStitch
