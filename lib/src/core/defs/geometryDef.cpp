// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "parse/json.hpp"
#include "core/transformGeoParams.hpp"
#include "common/angles.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"

namespace VideoStitch {
namespace Core {

GeometryDefinition::GeometryDefinition()
    : horizontalFocal(PTV_DEFAULT_INPUTDEF_HORIZONTAL_FOCAL),
      doesHaveVerticalFocal(PTV_DEFAULT_INPUTDEF_VERTICAL_FOCAL != 0.),
      verticalFocal(PTV_DEFAULT_INPUTDEF_VERTICAL_FOCAL),
      center_x(PTV_DEFAULT_INPUTDEF_LENSDIST_CENTER_X),
      center_y(PTV_DEFAULT_INPUTDEF_LENSDIST_CENTER_Y),
      distort_a(PTV_DEFAULT_INPUTDEF_LENSDISTA),
      distort_b(PTV_DEFAULT_INPUTDEF_LENSDISTB),
      distort_c(PTV_DEFAULT_INPUTDEF_LENSDISTC),
      distort_p1(PTV_DEFAULT_INPUTDEF_LENSDISTP1),
      distort_p2(PTV_DEFAULT_INPUTDEF_LENSDISTP2),
      distort_s1(PTV_DEFAULT_INPUTDEF_LENSDISTS1),
      distort_s2(PTV_DEFAULT_INPUTDEF_LENSDISTS2),
      distort_s3(PTV_DEFAULT_INPUTDEF_LENSDISTS3),
      distort_s4(PTV_DEFAULT_INPUTDEF_LENSDISTS4),
      distort_tau1(PTV_DEFAULT_INPUTDEF_LENSDISTTAU1),
      distort_tau2(PTV_DEFAULT_INPUTDEF_LENSDISTTAU2),
      yaw(PTV_DEFAULT_INPUTDEF_YAW),
      pitch(PTV_DEFAULT_INPUTDEF_PITCH),
      roll(PTV_DEFAULT_INPUTDEF_ROLL),
      doesHaveTranslation(PTV_DEFAULT_INPUTDEF_TRANS_X != 0. || PTV_DEFAULT_INPUTDEF_TRANS_Y != 0. ||
                          PTV_DEFAULT_INPUTDEF_TRANS_Z != 0.),
      translation_x(PTV_DEFAULT_INPUTDEF_TRANS_X),
      translation_y(PTV_DEFAULT_INPUTDEF_TRANS_Y),
      translation_z(PTV_DEFAULT_INPUTDEF_TRANS_Z),
      hasFovLoaded(false) {
  /*Default parameters if not created from ptv*/
}

GeometryDefinition::~GeometryDefinition() {}

bool GeometryDefinition::operator==(const GeometryDefinition& other) const {
#define FIELD_EQUAL(getter) (getter() == other.getter())

  bool res = FIELD_EQUAL(getDistortA);
  res &= FIELD_EQUAL(getDistortB);
  res &= FIELD_EQUAL(getDistortC);
  res &= FIELD_EQUAL(getDistortP1);
  res &= FIELD_EQUAL(getDistortP2);
  res &= FIELD_EQUAL(getDistortS1);
  res &= FIELD_EQUAL(getDistortS2);
  res &= FIELD_EQUAL(getDistortS3);
  res &= FIELD_EQUAL(getDistortS4);
  res &= FIELD_EQUAL(getDistortTau1);
  res &= FIELD_EQUAL(getDistortTau2);
  res &= FIELD_EQUAL(getCenterX);
  res &= FIELD_EQUAL(getCenterY);
  res &= FIELD_EQUAL(getYaw);
  res &= FIELD_EQUAL(getPitch);
  res &= FIELD_EQUAL(getRoll);
  res &= FIELD_EQUAL(getHorizontalFocal);
  res &= FIELD_EQUAL(getVerticalFocal);
  res &= FIELD_EQUAL(getTranslationX);
  res &= FIELD_EQUAL(getTranslationY);
  res &= FIELD_EQUAL(getTranslationZ);

  return res;
}

bool GeometryDefinition::hasSameExtrinsics(const GeometryDefinition& other) const {
  bool res = FIELD_EQUAL(getYaw);
  res &= FIELD_EQUAL(getPitch);
  res &= FIELD_EQUAL(getRoll);
  res &= FIELD_EQUAL(getTranslationX);
  res &= FIELD_EQUAL(getTranslationY);
  res &= FIELD_EQUAL(getTranslationZ);

  return res;
}

Status GeometryDefinition::applyDiff(const Ptv::Value& value, bool enforceMandatoryFields) {
#define PROPAGATE_NOK_NAMED(config_name, varName)                                                         \
  if (Parse::populateDouble("GeometryDefinition", value, config_name, varName, enforceMandatoryFields) != \
      Parse::PopulateResult_Ok) {                                                                         \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                                 \
            "No valid configuration for " config_name "' found, expected double"};                        \
  }
#define PROPAGATE_NOK(varName)                                                                         \
  if (Parse::populateDouble("GeometryDefinition", value, #varName, varName, enforceMandatoryFields) != \
      Parse::PopulateResult_Ok) {                                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                              \
            "No valid configuration for " #varName "' found, expected double"};                        \
  }
#define PROPAGATE_NOK_EXIST(varName)                                                         \
  {                                                                                          \
    auto res = Parse::populateDouble("GeometryDefinition", value, #varName, varName, false); \
    if (res != Parse::PopulateResult_Ok && res != Parse::PopulateResult_DoesNotExist) {      \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                  \
              "No valid configuration for " #varName "' found, expected double"};            \
    }                                                                                        \
  }
#define POPULATE_OPTIONAL(varName) Parse::populateDouble("GeometryDefinition", value, #varName, varName, false)

  if (!Parse::checkType("GeometryDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Invalid type for 'GeometryDefinition' configuration, expected object"};
  }

  PROPAGATE_NOK(yaw);
  PROPAGATE_NOK(pitch);
  PROPAGATE_NOK(roll);
  PROPAGATE_NOK(center_x);
  PROPAGATE_NOK(center_y);
  PROPAGATE_NOK(distort_a);
  PROPAGATE_NOK(distort_b);
  PROPAGATE_NOK(distort_c);

  PROPAGATE_NOK_EXIST(distort_p1);
  PROPAGATE_NOK_EXIST(distort_p2);
  PROPAGATE_NOK_EXIST(distort_s1);
  PROPAGATE_NOK_EXIST(distort_s2);
  PROPAGATE_NOK_EXIST(distort_s3);
  PROPAGATE_NOK_EXIST(distort_s4);
  PROPAGATE_NOK_EXIST(distort_tau1);
  PROPAGATE_NOK_EXIST(distort_tau2);

  if (Parse::populateDouble("GeometryDefinition", value, "horizontalFocal", horizontalFocal, false) !=
      Parse::PopulateResult_Ok) {
    PROPAGATE_NOK_NAMED("horizontalFov", horizontalFocal);
    this->hasFovLoaded = true;
  }

  if (Parse::populateDouble("GeometryDefinition", value, "verticalFocal", verticalFocal, false) ==
      Parse::PopulateResult_Ok) {
    doesHaveVerticalFocal = true;
  } else {
    doesHaveVerticalFocal = false;
    verticalFocal = 0.0;
  }

  Parse::PopulateResult resx, resy, resz;
  resx = POPULATE_OPTIONAL(translation_x);
  resy = POPULATE_OPTIONAL(translation_y);
  resz = POPULATE_OPTIONAL(translation_z);
  if (resx != Parse::PopulateResult_Ok || resy != Parse::PopulateResult_Ok || resz != Parse::PopulateResult_Ok) {
    translation_x = 0.0;
    translation_y = 0.0;
    translation_z = 0.0;
  }
  doesHaveTranslation = translation_x != 0.0 || translation_y != 0.0 || translation_z != 0.0;

#undef PROPAGATE_NOK_NAMED
#undef PROPAGATE_NOK
#undef PROPAGATE_NOK_EXIST
#undef POPULATE_OPTIONAL

  return Status::OK();
}

void GeometryDefinition::serialize(Ptv::Value& res) const {
  res.push("yaw", new Parse::JsonValue(getYaw()));
  res.push("pitch", new Parse::JsonValue(getPitch()));
  res.push("roll", new Parse::JsonValue(getRoll()));
  res.push("center_x", new Parse::JsonValue(getCenterX()));
  res.push("center_y", new Parse::JsonValue(getCenterY()));
  res.push("distort_a", new Parse::JsonValue(getDistortA()));
  res.push("distort_b", new Parse::JsonValue(getDistortB()));
  res.push("distort_c", new Parse::JsonValue(getDistortC()));
  // do not serialize advanced distortion parameter groups if they are zero
  if (distort_p1 != 0. || distort_p2 != 0.) {
    res.push("distort_p1", new Parse::JsonValue(getDistortP1()));
    res.push("distort_p2", new Parse::JsonValue(getDistortP2()));
  }
  if (distort_s1 != 0. || distort_s2 != 0. || distort_s3 != 0. || distort_s4 != 0.) {
    res.push("distort_s1", new Parse::JsonValue(getDistortS1()));
    res.push("distort_s2", new Parse::JsonValue(getDistortS2()));
    res.push("distort_s3", new Parse::JsonValue(getDistortS3()));
    res.push("distort_s4", new Parse::JsonValue(getDistortS4()));
  }
  if (distort_tau1 != 0. || distort_tau2 != 0.) {
    res.push("distort_tau1", new Parse::JsonValue(getDistortTau1()));
    res.push("distort_tau2", new Parse::JsonValue(getDistortTau2()));
  }
  res.push("horizontalFocal", new Parse::JsonValue(getHorizontalFocal()));

  if (doesHaveVerticalFocal) {
    res.push("verticalFocal", new Parse::JsonValue(getVerticalFocal()));
  }

  /* Save translation if values are non-zero, to keep foreward compatibility */
  if (doesHaveTranslation) {
    res.push("translation_x", new Parse::JsonValue(getTranslationX()));
    res.push("translation_y", new Parse::JsonValue(getTranslationY()));
    res.push("translation_z", new Parse::JsonValue(getTranslationZ()));
  }
}

#define GENGETTER(class, type, exportName, member) \
  type class ::get##exportName() const { return member; }

#define GENSETTER(class, type, exportName, member) \
  void class ::set##exportName(type member) { this->member = member; }

#define GENGETSETTER(class, type, exportName, member) \
  GENGETTER(class, type, exportName, member)          \
  GENSETTER(class, type, exportName, member)

GENGETSETTER(GeometryDefinition, double, DistortA, distort_a)
GENGETSETTER(GeometryDefinition, double, DistortB, distort_b)
GENGETSETTER(GeometryDefinition, double, DistortC, distort_c)
GENGETSETTER(GeometryDefinition, double, DistortP1, distort_p1)
GENGETSETTER(GeometryDefinition, double, DistortP2, distort_p2)
GENGETSETTER(GeometryDefinition, double, DistortS1, distort_s1)
GENGETSETTER(GeometryDefinition, double, DistortS2, distort_s2)
GENGETSETTER(GeometryDefinition, double, DistortS3, distort_s3)
GENGETSETTER(GeometryDefinition, double, DistortS4, distort_s4)
GENGETSETTER(GeometryDefinition, double, DistortTau1, distort_tau1)
GENGETSETTER(GeometryDefinition, double, DistortTau2, distort_tau2)
GENGETSETTER(GeometryDefinition, double, HorizontalFocal, horizontalFocal)

bool GeometryDefinition::hasDistortion() const { return hasRadialDistortion() || hasNonRadialDistortion(); }

bool GeometryDefinition::hasRadialDistortion() const {
  bool res = false;

  res |= getDistortA() != 0.;
  res |= getDistortB() != 0.;
  res |= getDistortC() != 0.;

  return res;
}

bool GeometryDefinition::hasNonRadialDistortion() const {
  bool res = false;

  res |= getDistortP1() != 0.;
  res |= getDistortP2() != 0.;
  res |= getDistortS1() != 0.;
  res |= getDistortS2() != 0.;
  res |= getDistortS3() != 0.;
  res |= getDistortS4() != 0.;
  res |= getDistortTau1() != 0.;
  res |= getDistortTau2() != 0.;

  return res;
}

void GeometryDefinition::convertLoadedFovToFocal(const InputDefinition& input) {
  if (this->hasFovLoaded) {
    this->horizontalFocal = TransformGeoParams::computeHorizontalScale(input, this->horizontalFocal);
    this->hasFovLoaded = true;
  }
}

double GeometryDefinition::getEstimatedHorizontalFov(const InputDefinition& input) const {
  return TransformGeoParams::computeFov(input, this->horizontalFocal);
}

void GeometryDefinition::setEstimatedHorizontalFov(const InputDefinition& input, double fov) {
  this->horizontalFocal = TransformGeoParams::computeHorizontalScale(input, fov);
}

double GeometryDefinition::getVerticalFocal() const {
  if (!this->doesHaveVerticalFocal) {
    return this->horizontalFocal;
  }

  return this->verticalFocal;
}

bool GeometryDefinition::hasVerticalFocal() const { return this->doesHaveVerticalFocal; }

void GeometryDefinition::setVerticalFocal(double focal) {
  this->doesHaveVerticalFocal = true;
  this->verticalFocal = focal;
}

GENGETSETTER(GeometryDefinition, double, CenterX, center_x)
GENGETSETTER(GeometryDefinition, double, CenterY, center_y)

GENGETSETTER(GeometryDefinition, double, Yaw, yaw)
GENGETSETTER(GeometryDefinition, double, Pitch, pitch)
GENGETSETTER(GeometryDefinition, double, Roll, roll)

#undef GENGETTER
#undef GENSETTER
#undef GENGETSETTER

bool GeometryDefinition::hasTranslation() const { return this->doesHaveTranslation; }

double GeometryDefinition::getTranslationX() const { return this->translation_x; }

void GeometryDefinition::setTranslationX(double translation_x) {
  this->translation_x = translation_x;
  doesHaveTranslation = translation_x != 0.0 || translation_y != 0.0 || translation_z != 0.0;
}

double GeometryDefinition::getTranslationY() const { return this->translation_y; }

void GeometryDefinition::setTranslationY(double translation_y) {
  this->translation_y = translation_y;
  doesHaveTranslation = translation_x != 0.0 || translation_y != 0.0 || translation_z != 0.0;
}

double GeometryDefinition::getTranslationZ() const { return this->translation_z; }

void GeometryDefinition::setTranslationZ(double translation_z) {
  this->translation_z = translation_z;
  doesHaveTranslation = translation_x != 0.0 || translation_y != 0.0 || translation_z != 0.0;
}

double GeometryDefinition::getGeoParamFromId(char id) const {
  switch (id) {
    case 'y':
      return getYaw();
    case 'p':
      return getPitch();
    case 'r':
      return getRoll();
    case 'f':
      return getHorizontalFocal();
    case 'a':
      return getDistortA();
    case 'b':
      return getDistortB();
    case 'c':
      return getDistortC();
    case 'd':
      return getCenterX();
    case 'e':
      return getCenterY();
    default:
      Logger::get(Logger::Error) << "Attempted to get a non-existing parameter id" << std::endl;
      return 0.0;
  }
}

void GeometryDefinition::setGeoParamFromId(char id, double val) {
  switch (id) {
    case 'y':
      setYaw(val);
      break;
    case 'p':
      setPitch(val);
      break;
    case 'r':
      setRoll(val);
      break;
    case 'f':
      setHorizontalFocal(val);
      break;
    case 'a':
      setDistortA(val);
      break;
    case 'b':
      setDistortB(val);
      break;
    case 'c':
      setDistortC(val);
      break;
    case 'd':
      setCenterX(val);
      break;
    case 'e':
      setCenterY(val);
      break;
    default:
      Logger::get(Logger::Error) << "Attempted to set a non-existing parameter id" << std::endl;
      break;
  }
}

void GeometryDefinition::applyGlobalOrientation(const Quaternion<double>& orientation) {
  /*Apply global orientation to input rotation angles*/
  Quaternion<double> q = Quaternion<double>::fromEulerZXY(degToRad(yaw), degToRad(pitch), degToRad(roll)) * orientation;

  q.toEuler(yaw, pitch, roll);

  /*Convert back to degrees*/
  yaw = radToDeg(yaw);
  pitch = radToDeg(pitch);
  roll = radToDeg(roll);
}

void GeometryDefinition::resetAllButFocal() {
  setYaw(0.0);
  setPitch(0.0);
  setRoll(0.0);
  resetDistortion();
  setTranslationX(0.0);
  setTranslationY(0.0);
  setTranslationZ(0.0);
}

void GeometryDefinition::resetExtrinsics() {
  setYaw(0.0);
  setPitch(0.0);
  setRoll(0.0);
  setTranslationX(0.0);
  setTranslationY(0.0);
  setTranslationZ(0.0);
}

void GeometryDefinition::resetDistortion() {
  setDistortA(0.0);
  setDistortB(0.0);
  setDistortC(0.0);
  setDistortP1(0.0);
  setDistortP2(0.0);
  setDistortS1(0.0);
  setDistortS2(0.0);
  setDistortS3(0.0);
  setDistortS4(0.0);
  setDistortTau1(0.0);
  setDistortTau2(0.0);
  setCenterX(0.0);
  setCenterY(0.0);
}

}  // namespace Core
}  // namespace VideoStitch
