// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "parse/json.hpp"

#include "libvideostitch/logging.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

StereoRigDefinition::Pimpl::Pimpl() : orientation(Landscape), geometry(Circular), diameter(0.0), ipd(0.0) {}

StereoRigDefinition::StereoRigDefinition() : pimpl(new Pimpl()) {}

StereoRigDefinition* StereoRigDefinition::clone() const {
  StereoRigDefinition* result = new StereoRigDefinition();

  result->setOrientation(getOrientation());
  result->setGeometry(getGeometry());
  result->setDiameter(getDiameter());
  result->setIPD(getIPD());
  for (size_t i = 0; i < pimpl->leftInputs.size(); ++i) {
    result->pimpl->leftInputs.push_back(pimpl->leftInputs[i]);
  }
  for (size_t i = 0; i < pimpl->rightInputs.size(); ++i) {
    result->pimpl->rightInputs.push_back(pimpl->rightInputs[i]);
  }
  return result;
}

bool StereoRigDefinition::operator==(const StereoRigDefinition& other) const {
#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getOrientation) && FIELD_EQUAL(getGeometry) && FIELD_EQUAL(getDiameter))) {
    return false;
  }
#undef FIELD_EQUAL
  for (size_t i = 0; i < pimpl->leftInputs.size(); ++i) {
    if (!(pimpl->leftInputs[i] == other.pimpl->leftInputs[i])) {
      return false;
    }
  }
  for (size_t i = 0; i < pimpl->rightInputs.size(); ++i) {
    if (!(pimpl->rightInputs[i] == other.pimpl->rightInputs[i])) {
      return false;
    }
  }

  return true;
}

bool StereoRigDefinition::validate(std::ostream& os) const {
  if (getGeometry() == Circular) {
    if (getDiameter() <= 0) {
      os << "diameter must be strictly positive." << std::endl;
      return false;
    }
    if (getDiameter() <= getIPD()) {
      os << "diameter must be strictly greater than the inter-pupillary distance." << std::endl;
      return false;
    }
    if (getLeftInputs() != getRightInputs()) {
      os << "for circular rigs, left and right inputs must be the same." << std::endl;
      return false;
    }
  }
  return true;
}

StereoRigDefinition::~StereoRigDefinition() { delete pimpl; }

StereoRigDefinition::Pimpl::~Pimpl() {}

double StereoRigDefinition::getDiameter() const { return pimpl->diameter; }

double StereoRigDefinition::getIPD() const { return pimpl->ipd; }

StereoRigDefinition::Orientation StereoRigDefinition::getOrientation() const { return pimpl->orientation; }

StereoRigDefinition::Geometry StereoRigDefinition::getGeometry() const { return pimpl->geometry; }

std::vector<int> StereoRigDefinition::getLeftInputs() const { return pimpl->leftInputs; }

std::vector<int> StereoRigDefinition::getRightInputs() const { return pimpl->rightInputs; }

void StereoRigDefinition::setDiameter(double diameter) { pimpl->diameter = diameter; }

void StereoRigDefinition::setIPD(double ipd) { pimpl->ipd = ipd; }

void StereoRigDefinition::setOrientation(Orientation orientation) { pimpl->orientation = orientation; }

void StereoRigDefinition::setGeometry(Geometry geometry) { pimpl->geometry = geometry; }

void StereoRigDefinition::setLeftInputs(const std::vector<int>& left) { pimpl->leftInputs = left; }

void StereoRigDefinition::setRightInputs(const std::vector<int>& right) { pimpl->rightInputs = right; }

const std::string StereoRigDefinition::getOrientationName(const Orientation orient) {
  switch (orient) {
    case StereoRigDefinition::Portrait:
      return "portrait";
    case StereoRigDefinition::Landscape:
      return "landscape";
    case StereoRigDefinition::Portrait_flipped:
      return "portrait_flipped";
    case StereoRigDefinition::Landscape_flipped:
      return "landscape_flipped";
    default:
      return "portrait";
  }
}

const std::string StereoRigDefinition::getGeometryName(const StereoRigDefinition::Geometry geom) {
  switch (geom) {
    case StereoRigDefinition::Circular:
      return "circular";
    case StereoRigDefinition::Polygonal:
      return "polygonal";
    default:
      return "circular";
  }
}

bool StereoRigDefinition::getOrientationFromName(const std::string& name, Orientation& orient) {
  if (!name.compare("portrait")) {
    orient = StereoRigDefinition::Portrait;
    return true;
  } else if (!name.compare("landscape")) {
    orient = StereoRigDefinition::Landscape;
    return true;
  } else if (!name.compare("portrait_flipped")) {
    orient = StereoRigDefinition::Portrait_flipped;
    return true;
  } else if (!name.compare("landscape_flipped")) {
    orient = StereoRigDefinition::Landscape_flipped;
    return true;
  }
  return false;
}

bool StereoRigDefinition::getGeometryFromName(const std::string& name, Geometry& geom) {
  if (!name.compare("circular")) {
    geom = StereoRigDefinition::Circular;
    return true;
  } else if (!name.compare("polygonal")) {
    geom = StereoRigDefinition::Polygonal;
    return true;
  }
  return false;
}

StereoRigDefinition* StereoRigDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("Rig", value, Ptv::Value::OBJECT)) {
    return nullptr;
  }
  std::unique_ptr<StereoRigDefinition> res(new StereoRigDefinition());
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return NULL;                          \
  }
  std::string tmp;
  PROPAGATE_NOK(Parse::populateString("StereoRigDefinition", value, "geometry", tmp, true));
  if (!getGeometryFromName(tmp, res->pimpl->geometry)) {
    return nullptr;
  }
  if (res->pimpl->geometry == StereoRigDefinition::Circular) {
    PROPAGATE_NOK(Parse::populateDouble("StereoRigDefinition", value, "diameter", res->pimpl->diameter, true));
    PROPAGATE_NOK(Parse::populateDouble("StereoRigDefinition", value, "ipd", res->pimpl->ipd, true));
    PROPAGATE_NOK(Parse::populateString("StereoRigDefinition", value, "orientation", tmp, true));
    if (!getOrientationFromName(tmp, res->pimpl->orientation)) {
      return nullptr;
    }
  }
#undef PROPAGATE_NOK
  {
    const Ptv::Value* val = value.has("left_inputs");
    if (val && val->isConvertibleTo(Ptv::Value::LIST)) {
      const std::vector<Ptv::Value*>& listValues = val->asList();
      for (auto v : listValues) {
        if (v->getType() != Ptv::Value::INT) {
          return nullptr;
        }
        res->pimpl->leftInputs.push_back((int)v->asInt());
      }
    }
  }
  {
    const Ptv::Value* val = value.has("right_inputs");
    if (val && val->isConvertibleTo(Ptv::Value::LIST)) {
      const std::vector<Ptv::Value*>& listValues = val->asList();
      for (auto v : listValues) {
        if (v->getType() != Ptv::Value::INT) {
          return nullptr;
        }
        res->pimpl->rightInputs.push_back((int)v->asInt());
      }
    }
  }

  std::stringstream errors;
  if (!res->validate(errors)) {
    Logger::get(Logger::Error) << errors.str();
    return nullptr;
  }
  return res.release();
}

Ptv::Value* StereoRigDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("diameter", new Parse::JsonValue(getDiameter()));
  res->push("ipd", new Parse::JsonValue(getIPD()));
  res->push("orientation", new Parse::JsonValue(getOrientationName(getOrientation())));
  res->push("geometry", new Parse::JsonValue(getGeometryName(getGeometry())));
  {
    Ptv::Value* jsonInputs = new Parse::JsonValue((void*)NULL);
    jsonInputs->asList();
    for (size_t i = 0; i < getLeftInputs().size(); ++i) {
      jsonInputs->asList().push_back(new Parse::JsonValue(getLeftInputs()[i]));
    }
    res->push("left_inputs", jsonInputs);
  }
  {
    Ptv::Value* jsonInputs = new Parse::JsonValue((void*)NULL);
    jsonInputs->asList();
    for (size_t i = 0; i < getRightInputs().size(); ++i) {
      jsonInputs->asList().push_back(new Parse::JsonValue(getRightInputs()[i]));
    }
    res->push("right_inputs", jsonInputs);
  }
  return res;
}
}  // namespace Core
}  // namespace VideoStitch
