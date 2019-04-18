// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "libvideostitch/overlayInputDef.hpp"
#include "parse/json.hpp"
#include "common/angles.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/cameraDef.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

GENCURVEFUNCTIONS(OverlayInputDefinition, Curve, ScaleCurve, scaleCurve, PTV_DEFAULT_OVERLAY_SCALE)
GENCURVEFUNCTIONS(OverlayInputDefinition, Curve, AlphaCurve, alphaCurve, PTV_DEFAULT_OVERLAY_ALPHA)
GENCURVEFUNCTIONS(OverlayInputDefinition, Curve, TransXCurve, transXCurve, PTV_DEFAULT_OVERLAY_TRANSX)
GENCURVEFUNCTIONS(OverlayInputDefinition, Curve, TransYCurve, transYCurve, PTV_DEFAULT_OVERLAY_TRANSY)
GENCURVEFUNCTIONS(OverlayInputDefinition, Curve, TransZCurve, transZCurve, PTV_DEFAULT_OVERLAY_TRANSZ)
GENCURVEFUNCTIONS(OverlayInputDefinition, QuaternionCurve, RotationCurve, rotationCurve, Quaternion<double>())

OverlayInputDefinition::Pimpl::Pimpl()
    : globalOrientationApplied(false),
      scaleCurve(new Curve(PTV_DEFAULT_OVERLAY_SCALE)),
      alphaCurve(new Curve(PTV_DEFAULT_OVERLAY_ALPHA)),
      transXCurve(new Curve(PTV_DEFAULT_OVERLAY_TRANSX)),
      transYCurve(new Curve(PTV_DEFAULT_OVERLAY_TRANSY)),
      transZCurve(new Curve(PTV_DEFAULT_OVERLAY_TRANSZ)),
      rotationCurve(new QuaternionCurve(Quaternion<double>())) {}

OverlayInputDefinition::OverlayInputDefinition() : ReaderInputDefinition(), pimpl(new Pimpl()) {}

OverlayInputDefinition* OverlayInputDefinition::create(const Ptv::Value& value, bool enforceMandatoryFields) {
  std::unique_ptr<OverlayInputDefinition> res(new OverlayInputDefinition());

  if (!res->applyDiff(value, enforceMandatoryFields).ok()) {
    return nullptr;
  }

  std::stringstream errors;
  if (!res->validate(errors)) {
    Logger::get(Logger::Error) << errors.str();
    return nullptr;
  }

  return res.release();
}

Status OverlayInputDefinition::applyDiff(const Ptv::Value& value, bool enforceMandatoryFields) {
  Status stat;

  // Make sure value is an object.
  if (!Parse::checkType("OverlayInputDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Could not find valid 'OverlayInputDefinition', expected object type"};
  }

  stat = ReaderInputDefinition::applyDiff(value, enforceMandatoryFields);
  FAIL_RETURN(stat);

#define POPULATE_CURVE_PROPAGATE_WRONGTYPE(config_name, varName, varType)                            \
  {                                                                                                  \
    const Ptv::Value* var = value.has(config_name);                                                  \
    if (var) {                                                                                       \
      varType* curve = varType::create(*var);                                                        \
      pimpl->varName.reset(curve);                                                                   \
    } else {                                                                                         \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                          \
              "Invalid type for '" config_name "' in OverlayInputDefinition, expected curve value"}; \
    }                                                                                                \
  }
#define POPULATE_BOOL_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                       \
  if (Parse::populateBool("OverlayInputDefinition", value, config_name, varName, shouldEnforce) ==   \
      Parse::PopulateResult_WrongType) {                                                             \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                            \
            "Invalid type for '" config_name "' in OverlayInputDefinition, expected boolean value"}; \
  }
#define POPULATE_INT_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                        \
  if (Parse::populateInt("OverlayInputDefinition", value, config_name, varName, shouldEnforce) ==    \
      Parse::PopulateResult_WrongType) {                                                             \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                            \
            "Invalid type for '" config_name "' in OverlayInputDefinition, expected integer value"}; \
  }
  POPULATE_BOOL_PROPAGATE_WRONGTYPE("global_orientation_applied", pimpl->globalOrientationApplied, false);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("scale", scaleCurve, Curve);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("alpha", alphaCurve, Curve);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("transX", transXCurve, Curve);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("transY", transYCurve, Curve);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("transZ", transZCurve, Curve);
  POPULATE_CURVE_PROPAGATE_WRONGTYPE("rotation", rotationCurve, QuaternionCurve);
#undef POPULATE_CURVE_PROPAGATE_WRONGTYPE
#undef POPULATE_BOOL_PROPAGATE_WRONGTYPE
#undef POPULATE_INT_PROPAGATE_WRONGTYPE

  return stat;
};

bool OverlayInputDefinition::validate(std::ostream& os) const {
  if (!ReaderInputDefinition::validate(os)) {
    return false;
  }

#define OS_MESSAGE(varName, infValue, supValue)                                                          \
  {                                                                                                      \
    os << #varName << " must be in the interval [" << infValue << ", " << supValue << "]." << std::endl; \
    return false;                                                                                        \
  }
#define VALIDATE_CURVE_VALUE(varName, infValue, supValue)                                                  \
  {                                                                                                        \
    auto spline = get##varName().splines();                                                                \
    if (spline) {                                                                                          \
      while (spline) {                                                                                     \
        if (spline->end.v < infValue || spline->end.v > supValue) {                                        \
          OS_MESSAGE(varName, infValue, supValue)                                                          \
        } else {                                                                                           \
          spline = spline->next;                                                                           \
        }                                                                                                  \
      }                                                                                                    \
    } else {                                                                                               \
      if (get##varName().getConstantValue() <= infValue || get##varName().getConstantValue() > supValue) { \
        OS_MESSAGE(varName, infValue, supValue)                                                            \
      }                                                                                                    \
    }                                                                                                      \
  }
  VALIDATE_CURVE_VALUE(ScaleCurve, 0.0, 1.0);
  VALIDATE_CURVE_VALUE(AlphaCurve, 0.0, 1.0);
#undef OS_MESSAGE
#undef VALIDATE_CURVE_VALUE

  return true;
}

OverlayInputDefinition* OverlayInputDefinition::clone() const {
  OverlayInputDefinition* result = new OverlayInputDefinition();
  cloneTo(result);

#define AUTO_CURVE_COPY(curve) result->replace##curve(get##curve().clone())
  result->pimpl->globalOrientationApplied = pimpl->globalOrientationApplied;
  AUTO_CURVE_COPY(ScaleCurve);
  AUTO_CURVE_COPY(TransXCurve);
  AUTO_CURVE_COPY(TransYCurve);
  AUTO_CURVE_COPY(TransZCurve);
  AUTO_CURVE_COPY(RotationCurve);
  AUTO_CURVE_COPY(AlphaCurve);
#undef AUTO_CURVE_COPY

  return result;
}

bool OverlayInputDefinition::operator==(const OverlayInputDefinition& other) const {
  if (!ReaderInputDefinition::operator==(other)) {
    return false;
  }

#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getGlobalOrientationApplied) && FIELD_EQUAL(getScaleCurve) && FIELD_EQUAL(getTransXCurve) &&
        FIELD_EQUAL(getTransYCurve) && FIELD_EQUAL(getTransZCurve) && FIELD_EQUAL(getRotationCurve) &&
        FIELD_EQUAL(getAlphaCurve))) {
    return false;
  }

  return true;
#undef FIELD_EQUAL
}

OverlayInputDefinition::~OverlayInputDefinition() { delete pimpl; }

OverlayInputDefinition::Pimpl::~Pimpl() {}

Ptv::Value* OverlayInputDefinition::serialize() const {
  Ptv::Value* res = ReaderInputDefinition::serialize();

  res->push("global_orientation_applied", new Parse::JsonValue(getGlobalOrientationApplied()));
  res->push("scale", getScaleCurve().serialize());
  res->push("alpha", getAlphaCurve().serialize());
  res->push("transX", getTransXCurve().serialize());
  res->push("transY", getTransYCurve().serialize());
  res->push("transZ", getTransZCurve().serialize());
  res->push("rotation", getRotationCurve().serialize());

  return res;
}

/************ Getters and Setters **********/
bool OverlayInputDefinition::getGlobalOrientationApplied() const { return pimpl->globalOrientationApplied; }

void OverlayInputDefinition::setGlobalOrientationApplied(const bool status) {
  pimpl->globalOrientationApplied = status;
}

}  // namespace Core
}  // namespace VideoStitch
