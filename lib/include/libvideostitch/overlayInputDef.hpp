// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OVERLAY_INPUT_DEF_HPP_
#define OVERLAY_INPUT_DEF_HPP_

#include "config.hpp"
#include "curves.hpp"
#include "status.hpp"
#include "readerInputDef.hpp"

#include <string>

namespace VideoStitch {

class ThreadSafeOstream;

namespace Ptv {
class Value;
}

namespace Core {
/**
 * @brief A Overlay setup representation class.
 */
class VS_EXPORT OverlayInputDefinition : public ReaderInputDefinition {
 public:
  virtual ~OverlayInputDefinition();

  /**
   * Clones an OverlayInputDefinition java-style.
   * @return A similar OverlayInputDefinition. Ownership is given to the caller.
   */
  virtual OverlayInputDefinition* clone() const;

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed OverlayInputDefinition, or NULL on error.
   */
  static OverlayInputDefinition* create(const Ptv::Value& value, bool enforceMandatoryFields = true);

  virtual Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  virtual bool operator==(const OverlayInputDefinition& other) const;

  /**
   * Validates that the panorama makes sense.
   * @param os The sink for error messages.
   * @return false in case of failure.
   */
  virtual bool validate(std::ostream& os) const;

  /**
   * @brief Get the overlay GlobalOrientationApplied status
   * @return a status as bool
   */
  virtual bool getGlobalOrientationApplied() const;

  /**
   * @brief Set the GlobalOrientationApplied status
   * @param status the new value to set
   */
  virtual void setGlobalOrientationApplied(const bool status);

  DECLARE_CURVE(ScaleCurve, double)
  DECLARE_CURVE(TransXCurve, double)
  DECLARE_CURVE(TransYCurve, double)
  DECLARE_CURVE(TransZCurve, double)
  DECLARE_CURVE(RotationCurve, Quaternion<double>)
  DECLARE_CURVE(AlphaCurve, double)

 protected:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  OverlayInputDefinition();

  /**
   * Disabled, use clone()
   */
  OverlayInputDefinition(const OverlayInputDefinition&) = delete;

  /**
   * Disabled, use clone()
   */
  OverlayInputDefinition& operator=(const OverlayInputDefinition&) = delete;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param diff Input diff.
   */
  Status applyDiff(const Ptv::Value& diff, bool enforceMandatoryFields);

 private:
  class Pimpl;
  Pimpl* const pimpl;

  // keep the compiler happy
  using ReaderInputDefinition::operator==;
};
}  // namespace Core
}  // namespace VideoStitch

#endif
