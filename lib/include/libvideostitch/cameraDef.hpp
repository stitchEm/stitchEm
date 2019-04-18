// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "object.hpp"
#include <vector>
#include <string>

#include "inputDef.hpp"

namespace VideoStitch {
namespace Ptv {
class Value;
}
namespace Core {

class RigDefinition;

/**
Probabilist float value with two first modes
*/
typedef struct {
  /**
  Mean value
  */
  float mean;
  /**
  Variance value (second mode)
  */
  float variance;
} NormalFloat;

/**
Probabilist double value with two first modes
*/
typedef struct {
  /**
  Mean value
  */
  double mean;
  /**
  Variance value (second mode)
  */
  double variance;
} NormalDouble;

/**
 * @brief A camera model representation class.
 */
class VS_EXPORT CameraDefinition : public Ptv::Object {
 public:
  ~CameraDefinition();

  /**
   * Clones a camera java-style.
   * @return A similar camera. Ownership is given to the caller.
   */
  CameraDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed camera, or NULL on error.
   */
  static CameraDefinition* create(const Ptv::Value& value);

  /**
   * Build from a Ptv::Value.
   * @param name camera name.
   * @return The parsed camera, or NULL on error.
   */
  static CameraDefinition* createDefault(const std::string& name);

  Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  bool operator==(const CameraDefinition& other) const;

  /**
   * Validate that the rig makes sense.
   * @param os The sink for error messages.
   * @return false of failure.
   */
  bool validate(std::ostream& os) const;

  /**
   * Get the Camera definition name
   * @return a name as string
   */
  std::string getName() const;

  /**
   * Set the camera definition name
   * @param name the new value to set
   */
  void setName(const std::string& name);

  /**
   * Get the Camera input type
   * @return a input type
   */
  InputDefinition::Format getType() const;

  /**
   * Set the camera input type
   * @param val the new value to set
   */
  void setType(const InputDefinition::Format& val);

  /**
   * Get the size of the horizontal dimension
   * @return the width
   */
  size_t getWidth() const;

  /**
   * Set the new horizontal width in pixels
   * @param val the new value to set
   */
  void setWidth(size_t val);

  /**
   * Get the size of the horizontal dimension
   * @return the width
   */
  size_t getHeight() const;

  /**
   * Set the new vertical height in pixels
   * @param val the new value to set
   */
  void setHeight(size_t val);

  /**
   * Get the focal in pixels for the horizontal dimension
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getFu() const;

  /**
   * Set the new horizontal focal in pixels
   * @param val the new value to set
   */
  void setFu(const NormalDouble& val);

  /**
   * Set the focal using the fov value
   * Assume the caemra type is set
   * @param fov the fov to use as basic value
   */
  void setFov(const NormalDouble& fov);

  /**
   * Get the focal in pixels for the vertical dimension
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getFv() const;

  /**
   * Set the new vertical focal in pixels
   * @param val the new value to set
   */
  void setFv(const NormalDouble& val);

  /**
   * Get the center coordinate in pixels for the horizontal dimension
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getCu() const;

  /**
   * Set the new horizontal center in pixels
   * @param val the new value to set
   */
  void setCu(const NormalDouble& val);

  /**
   * Get the center coordinate in pixels for the vertical dimension
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getCv() const;

  /**
   * Set the new vertical center in pixels
   * @param val the new value to set
   */
  void setCv(const NormalDouble& val);

  /**
   * Get the First distortion parameter
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getDistortionA() const;

  /**
   * Set the new first distortion value
   * @param val the new value to set
   */
  void setDistortionA(const NormalDouble& val);

  /**
   * Get the second distortion parameter
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getDistortionB() const;

  /**
   * Set the new second distortion value
   * @param val the new value to set
   */
  void setDistortionB(const NormalDouble& val);

  /**
   * Get the third distortion parameter
   * @return an NormalDouble (mean+variance)
   */
  NormalDouble getDistortionC() const;

  /**
   * Set the new third distortion value
   * @param val the new value to set
   */
  void setDistortionC(const NormalDouble& val);

 public:
  /**
  Constant value for max bound value
  */
  static const double max_fu_variance;

  /**
  Constant value for max bound value
  */
  static const double max_fv_variance;

  /**
  Constant value for max bound value
  */
  static const double max_cu_variance;

  /**
  Constant value for max bound value
  */
  static const double max_cv_variance;

  /**
  Constant value for max bound value
  */
  static const double max_distorta_variance;

  /**
  Constant value for max bound value
  */
  static const double max_distortb_variance;

  /**
  Constant value for max bound value
  */
  static const double max_distortc_variance;

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  CameraDefinition();

  /**
   * Disabled, use clone()
   */
  CameraDefinition(const CameraDefinition&) = delete;
  /**
   * Disabled, use clone()
   */
  CameraDefinition& operator=(const CameraDefinition&) = delete;

 private:
  class Pimpl;
  Pimpl* const pimpl;

  friend class RigDefinition;
};
}  // namespace Core
}  // namespace VideoStitch
