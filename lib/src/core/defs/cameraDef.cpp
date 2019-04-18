// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "parse/json.hpp"
#include "core/transformGeoParams.hpp"

#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Core {

namespace {
bool operator!=(NormalDouble const& a, NormalDouble const& b) {
  return (a.mean == b.mean) && (a.variance == b.variance);
}
}  // namespace

const double CameraDefinition::max_fu_variance = 1e12;
const double CameraDefinition::max_fv_variance = 1e12;
const double CameraDefinition::max_cu_variance = 1e8;
const double CameraDefinition::max_cv_variance = 1e8;
const double CameraDefinition::max_distorta_variance = 1e4;
const double CameraDefinition::max_distortb_variance = 1e4;
const double CameraDefinition::max_distortc_variance = 1e4;

CameraDefinition::Pimpl::Pimpl() : name("") {
  width = 1;
  height = 1;
  fu.mean = 0.0;
  fu.variance = CameraDefinition::max_fu_variance;
  fv.mean = 0.0;
  fv.variance = CameraDefinition::max_fv_variance;
  cu.mean = 0.0;
  cu.variance = CameraDefinition::max_cu_variance;
  cv.mean = 0.0;
  cv.variance = CameraDefinition::max_cv_variance;
  distortion[0].mean = 0.0;
  distortion[0].variance = CameraDefinition::max_distorta_variance;
  distortion[1].mean = 0.0;
  distortion[1].variance = CameraDefinition::max_distortb_variance;
  distortion[2].mean = 0.0;
  distortion[2].variance = CameraDefinition::max_distortc_variance;
  format = InputDefinition::Format::FullFrameFisheye_Opt;
}

CameraDefinition::CameraDefinition() : pimpl(new Pimpl()) {}

CameraDefinition* CameraDefinition::clone() const {
  CameraDefinition* result = new CameraDefinition();

  result->setName(getName());
  result->setType(getType());

  result->setWidth(getWidth());
  result->setHeight(getHeight());

  result->setFu(getFu());
  result->setFv(getFv());
  result->setCu(getCu());
  result->setCv(getCv());

  result->setDistortionA(getDistortionA());
  result->setDistortionB(getDistortionB());
  result->setDistortionC(getDistortionC());

  return result;
}

bool CameraDefinition::operator==(const CameraDefinition& other) const {
  if (getWidth() != other.getWidth()) return false;
  if (getHeight() != other.getHeight()) return false;

  if (getFu() != other.getFu()) return false;
  if (getFv() != other.getFv()) return false;
  if (getCu() != other.getCu()) return false;
  if (getCv() != other.getCv()) return false;

  if (getDistortionA() != other.getDistortionA()) return false;
  if (getDistortionB() != other.getDistortionB()) return false;
  if (getDistortionC() != other.getDistortionC()) return false;

  if (getName() != other.getName()) return false;
  if (getType() != other.getType()) return false;

  return true;
}

bool CameraDefinition::validate(std::ostream& os) const {
  if (getName() == "") {
    os << "Invalid name" << std::endl;
    return false;
  }

  if (getFu().mean < 0.0) {
    os << "Invalid horizontal focal" << std::endl;
    return false;
  }

  if (getFv().mean < 0.0) {
    os << "Invalid vertical focal" << std::endl;
    return false;
  }

  if (getFu().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getFv().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getCu().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getCv().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getDistortionA().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getDistortionB().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  if (getDistortionC().variance < 0.0) {
    os << "Invalid variance value" << std::endl;
  }

  return true;
}

CameraDefinition::~CameraDefinition() { delete pimpl; }

CameraDefinition::Pimpl::~Pimpl() {}

CameraDefinition* CameraDefinition::createDefault(const std::string& name) {
  std::unique_ptr<CameraDefinition> res(new CameraDefinition());

  res->pimpl->name = name;

  return res.release();
}

CameraDefinition* CameraDefinition::create(const Ptv::Value& value) {
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return nullptr;                       \
  }

  if (!Parse::checkType("Camera", value, Ptv::Value::OBJECT)) {
    return nullptr;
  }

  std::unique_ptr<CameraDefinition> res(new CameraDefinition());
  std::string stype;

  PROPAGATE_NOK(Parse::populateString("CameraDefinition", value, "name", res->pimpl->name, true));
  PROPAGATE_NOK(Parse::populateString("CameraDefinition", value, "format", stype, true));

  if (!InputDefinition::getFormatFromName(stype, res->pimpl->format)) {
    return nullptr;
  }

  PROPAGATE_NOK(Parse::populateInt("CameraDefinition", value, "width", res->pimpl->width, true));
  PROPAGATE_NOK(Parse::populateInt("CameraDefinition", value, "height", res->pimpl->height, true));

  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "fu_mean", res->pimpl->fu.mean, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "fv_mean", res->pimpl->fv.mean, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "cu_mean", res->pimpl->cu.mean, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "cv_mean", res->pimpl->cv.mean, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distorta_mean", res->pimpl->distortion[0].mean, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distortb_mean", res->pimpl->distortion[1].mean, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distortc_mean", res->pimpl->distortion[2].mean, true));

  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "fu_variance", res->pimpl->fu.variance, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "fv_variance", res->pimpl->fv.variance, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "cu_variance", res->pimpl->cu.variance, true));
  PROPAGATE_NOK(Parse::populateDouble("CameraDefinition", value, "cv_variance", res->pimpl->cv.variance, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distorta_variance", res->pimpl->distortion[0].variance, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distortb_variance", res->pimpl->distortion[1].variance, true));
  PROPAGATE_NOK(
      Parse::populateDouble("CameraDefinition", value, "distortc_variance", res->pimpl->distortion[2].variance, true));

  if (res->pimpl->fu.variance > CameraDefinition::max_fu_variance)
    res->pimpl->fu.variance = CameraDefinition::max_fu_variance;
  if (res->pimpl->fv.variance > CameraDefinition::max_fv_variance)
    res->pimpl->fv.variance = CameraDefinition::max_fv_variance;
  if (res->pimpl->cu.variance > CameraDefinition::max_cu_variance)
    res->pimpl->cu.variance = CameraDefinition::max_cu_variance;
  if (res->pimpl->cv.variance > CameraDefinition::max_cv_variance)
    res->pimpl->cv.variance = CameraDefinition::max_cv_variance;
  if (res->pimpl->distortion[0].variance > CameraDefinition::max_distorta_variance)
    res->pimpl->distortion[0].variance = CameraDefinition::max_distorta_variance;
  if (res->pimpl->distortion[1].variance > CameraDefinition::max_distortb_variance)
    res->pimpl->distortion[1].variance = CameraDefinition::max_distortb_variance;
  if (res->pimpl->distortion[2].variance > CameraDefinition::max_distortc_variance)
    res->pimpl->distortion[2].variance = CameraDefinition::max_distortc_variance;

#undef PROPAGATE_NOK

  return res.release();
}

Ptv::Value* CameraDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();

  std::string typestr = InputDefinition::getFormatName(getType());

  res->push("name", new Parse::JsonValue(getName()));
  res->push("format", new Parse::JsonValue(typestr));

  res->push("width", new Parse::JsonValue((int)getWidth()));
  res->push("height", new Parse::JsonValue((int)getHeight()));

  res->push("fu_mean", new Parse::JsonValue(getFu().mean));
  res->push("fu_variance", new Parse::JsonValue(getFu().variance));

  res->push("fv_mean", new Parse::JsonValue(getFv().mean));
  res->push("fv_variance", new Parse::JsonValue(getFv().variance));

  res->push("cu_mean", new Parse::JsonValue(getCu().mean));
  res->push("cu_variance", new Parse::JsonValue(getCu().variance));

  res->push("cv_mean", new Parse::JsonValue(getCv().mean));
  res->push("cv_variance", new Parse::JsonValue(getCv().variance));

  res->push("distorta_mean", new Parse::JsonValue(getDistortionA().mean));
  res->push("distorta_variance", new Parse::JsonValue(getDistortionA().variance));

  res->push("distortb_mean", new Parse::JsonValue(getDistortionB().mean));
  res->push("distortb_variance", new Parse::JsonValue(getDistortionB().variance));

  res->push("distortc_mean", new Parse::JsonValue(getDistortionC().mean));
  res->push("distortc_variance", new Parse::JsonValue(getDistortionC().variance));

  return res;
}

/************ Getters and Setters **********/

std::string CameraDefinition::getName() const { return pimpl->name; }

void CameraDefinition::setName(const std::string& val) { pimpl->name = val; }

InputDefinition::Format CameraDefinition::getType() const { return pimpl->format; }

void CameraDefinition::setType(const InputDefinition::Format& val) { pimpl->format = val; }

NormalDouble CameraDefinition::getFu() const { return pimpl->fu; }

void CameraDefinition::setFu(const NormalDouble& val) { pimpl->fu = val; }

size_t CameraDefinition::getWidth() const { return pimpl->width; }

void CameraDefinition::setWidth(size_t val) { pimpl->width = val; }

size_t CameraDefinition::getHeight() const { return pimpl->height; }

void CameraDefinition::setHeight(size_t val) { pimpl->height = val; }

void CameraDefinition::setFov(const NormalDouble& fov) {
  pimpl->fu.mean = TransformGeoParams::computeInputScale(pimpl->format, pimpl->width, (float)fov.mean);
  pimpl->fu.variance = fov.variance;

  pimpl->fv.mean = pimpl->fu.mean;
  pimpl->fv.variance = pimpl->fu.variance;
}

NormalDouble CameraDefinition::getFv() const { return pimpl->fv; }

void CameraDefinition::setFv(const NormalDouble& val) { pimpl->fv = val; }

NormalDouble CameraDefinition::getCu() const { return pimpl->cu; }

void CameraDefinition::setCu(const NormalDouble& val) { pimpl->cu = val; }

NormalDouble CameraDefinition::getCv() const { return pimpl->cv; }

void CameraDefinition::setCv(const NormalDouble& val) { pimpl->cv = val; }

NormalDouble CameraDefinition::getDistortionA() const { return pimpl->distortion[0]; }

void CameraDefinition::setDistortionA(const NormalDouble& val) { pimpl->distortion[0] = val; }

NormalDouble CameraDefinition::getDistortionB() const { return pimpl->distortion[1]; }

void CameraDefinition::setDistortionB(const NormalDouble& val) { pimpl->distortion[1] = val; }

NormalDouble CameraDefinition::getDistortionC() const { return pimpl->distortion[2]; }

void CameraDefinition::setDistortionC(const NormalDouble& val) { pimpl->distortion[2] = val; }

}  // namespace Core
}  // namespace VideoStitch
