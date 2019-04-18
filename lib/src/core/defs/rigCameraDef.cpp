// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "parse/json.hpp"
#include "common/angles.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/rigCameraDef.hpp"

namespace VideoStitch {
namespace Core {

namespace {
bool operator!=(NormalDouble const& a, NormalDouble const& b) {
  return (a.mean == b.mean) && (a.variance == b.variance);
}
}  // namespace

const double RigCameraDefinition::max_yaw_variance = 4.0 * M_PI * M_PI;
const double RigCameraDefinition::max_pitch_variance = 4.0 * M_PI * M_PI;
const double RigCameraDefinition::max_roll_variance = 4.0 * M_PI * M_PI;
const double RigCameraDefinition::max_translation_x_variance = 100;
const double RigCameraDefinition::max_translation_y_variance = 100;
const double RigCameraDefinition::max_translation_z_variance = 100;

RigCameraDefinition::RigCameraDefinition() : pimpl(new Pimpl()) {}

RigCameraDefinition::~RigCameraDefinition() { delete pimpl; }

RigCameraDefinition::RigCameraDefinition(const RigCameraDefinition& other) : pimpl(new Pimpl(*other.pimpl)) {}

RigCameraDefinition::Pimpl::Pimpl() {
  yaw.mean = 0.0;
  yaw.variance = RigCameraDefinition::max_yaw_variance;
  pitch.mean = 0.0;
  pitch.variance = RigCameraDefinition::max_pitch_variance;
  roll.mean = 0.0;
  roll.variance = RigCameraDefinition::max_roll_variance;
  translation_x.mean = 0.0;
  translation_x.variance = RigCameraDefinition::max_translation_x_variance;
  translation_y.mean = 0.0;
  translation_y.variance = RigCameraDefinition::max_translation_y_variance;
  translation_z.mean = 0.0;
  translation_z.variance = RigCameraDefinition::max_translation_z_variance;
}

RigCameraDefinition::Pimpl::Pimpl(const RigCameraDefinition::Pimpl& other)
    : yaw(other.yaw),
      pitch(other.pitch),
      roll(other.roll),
      translation_x(other.translation_x),
      translation_y(other.translation_y),
      translation_z(other.translation_z),
      camera(other.camera) {}

RigCameraDefinition::Pimpl::~Pimpl() {}

RigCameraDefinition& RigCameraDefinition::operator=(const RigCameraDefinition& other) {
  if (&other != this) {
    pimpl->yaw = other.pimpl->yaw;
    pimpl->pitch = other.pimpl->pitch;
    pimpl->roll = other.pimpl->roll;
    pimpl->translation_x = other.pimpl->translation_x;
    pimpl->translation_y = other.pimpl->translation_y;
    pimpl->translation_z = other.pimpl->translation_z;
    pimpl->camera = other.pimpl->camera;
  }
  return *this;
}

bool RigCameraDefinition::operator==(const RigCameraDefinition& other) const {
  if (pimpl->yaw != other.pimpl->yaw) return false;
  if (pimpl->pitch != other.pimpl->pitch) return false;
  if (pimpl->roll != other.pimpl->roll) return false;
  if (pimpl->translation_x != other.pimpl->translation_x) return false;
  if (pimpl->translation_y != other.pimpl->translation_y) return false;
  if (pimpl->translation_z != other.pimpl->translation_z) return false;
  if (!(*pimpl->camera == *other.pimpl->camera)) return false;

  return true;
}

bool RigCameraDefinition::deserialize(const std::map<std::string, std::shared_ptr<CameraDefinition> >& camerasmap,
                                      const Ptv::Value& value) {
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return false;                         \
  }
#define PROPAGATE_NOK_EXIST(call)                                                     \
  res = call;                                                                         \
  if (res != Parse::PopulateResult_Ok && res != Parse::PopulateResult_DoesNotExist) { \
    return false;                                                                     \
  }

  if (!Parse::checkType("RigCamera", value, Ptv::Value::OBJECT)) {
    return false;
  }

  Parse::PopulateResult res;

  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "yaw_mean", pimpl->yaw.mean, true));
  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "pitch_mean", pimpl->pitch.mean, true));
  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "roll_mean", pimpl->roll.mean, true));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_x_mean", pimpl->translation_x.mean, false));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_y_mean", pimpl->translation_y.mean, false));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_z_mean", pimpl->translation_z.mean, false));
  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "yaw_variance", pimpl->yaw.variance, true));
  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "pitch_variance", pimpl->pitch.variance, true));
  PROPAGATE_NOK(Parse::populateDouble("RigCamera", value, "roll_variance", pimpl->roll.variance, true));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_x_variance", pimpl->translation_x.variance, false));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_y_variance", pimpl->translation_y.variance, false));
  PROPAGATE_NOK_EXIST(
      Parse::populateDouble("RigCamera", value, "translation_z_variance", pimpl->translation_z.variance, false));

  std::string angle_unit;
  if (Parse::populateString("RigCamera", value, "angle_unit", angle_unit, false) == Parse::PopulateResult_Ok &&
      angle_unit.compare("degrees") == 0) {
    // convert the rotations from degrees to radians
    pimpl->yaw.mean *= M_PI / 180;
    pimpl->pitch.mean *= M_PI / 180;
    pimpl->roll.mean *= M_PI / 180;
    pimpl->yaw.variance *= (M_PI / 180) * (M_PI / 180);
    pimpl->pitch.variance *= (M_PI / 180) * (M_PI / 180);
    pimpl->roll.variance *= (M_PI / 180) * (M_PI / 180);
  }

  if (pimpl->yaw.variance > RigCameraDefinition::max_yaw_variance)
    pimpl->yaw.variance = RigCameraDefinition::max_yaw_variance;
  if (pimpl->pitch.variance > RigCameraDefinition::max_pitch_variance)
    pimpl->pitch.variance = RigCameraDefinition::max_pitch_variance;
  if (pimpl->roll.variance > RigCameraDefinition::max_roll_variance)
    pimpl->roll.variance = RigCameraDefinition::max_roll_variance;

  std::string camname;
  PROPAGATE_NOK(Parse::populateString("RigCamera", value, "camera", camname, true));

  std::map<std::string, std::shared_ptr<CameraDefinition> >::const_iterator it = camerasmap.find(camname);
  if (it == camerasmap.end()) return false;

  pimpl->camera = camerasmap.at(camname);

#undef PROPAGATE_NOK
#undef PROPAGATE_NOK_EXIST

  return true;
}

Ptv::Value* RigCameraDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();

  res->push("angle_unit", new Parse::JsonValue("degrees"));
  res->push("yaw_mean", new Parse::JsonValue(radToDeg(pimpl->yaw.mean)));
  res->push("pitch_mean", new Parse::JsonValue(radToDeg(pimpl->pitch.mean)));
  res->push("roll_mean", new Parse::JsonValue(radToDeg(pimpl->roll.mean)));
  res->push("translation_x_mean", new Parse::JsonValue(pimpl->translation_x.mean));
  res->push("translation_y_mean", new Parse::JsonValue(pimpl->translation_y.mean));
  res->push("translation_z_mean", new Parse::JsonValue(pimpl->translation_z.mean));
  res->push("yaw_variance", new Parse::JsonValue(radToDeg(radToDeg(pimpl->yaw.variance))));
  res->push("pitch_variance", new Parse::JsonValue(radToDeg(radToDeg(pimpl->pitch.variance))));
  res->push("roll_variance", new Parse::JsonValue(radToDeg(radToDeg(pimpl->roll.variance))));
  res->push("translation_x_variance", new Parse::JsonValue(pimpl->translation_x.variance));
  res->push("translation_y_variance", new Parse::JsonValue(pimpl->translation_y.variance));
  res->push("translation_z_variance", new Parse::JsonValue(pimpl->translation_z.variance));

  if (pimpl->camera.get()) {
    res->push("camera", new Parse::JsonValue(pimpl->camera->getName()));
  } else {
    res->push("camera", new Parse::JsonValue("unknown"));
  }

  return res;
}

/************ Getters and Setters **********/

GENGETREFSETTER(RigCameraDefinition, NormalDouble, YawRadians, yaw)
GENGETREFSETTER(RigCameraDefinition, NormalDouble, PitchRadians, pitch)
GENGETREFSETTER(RigCameraDefinition, NormalDouble, RollRadians, roll)
GENGETREFSETTER(RigCameraDefinition, NormalDouble, TranslationX, translation_x)
GENGETREFSETTER(RigCameraDefinition, NormalDouble, TranslationY, translation_y)
GENGETREFSETTER(RigCameraDefinition, NormalDouble, TranslationZ, translation_z)

void RigCameraDefinition::setCamera(const std::shared_ptr<CameraDefinition>& value) { pimpl->camera = value; }

std::shared_ptr<CameraDefinition> RigCameraDefinition::getCamera() const { return pimpl->camera; }

bool RigCameraDefinition::fillGeometryDefinition(GeometryDefinition& /*def*/) const { return true; }

}  // namespace Core
}  // namespace VideoStitch
