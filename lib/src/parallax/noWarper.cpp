// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "noWarper.hpp"

#include "parse/json.hpp"

#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageWarperFactory> NoWarper::Factory::parse(const Ptv::Value&) {
  return Potential<ImageWarperFactory>(new NoWarper::Factory());
}

Ptv::Value* NoWarper::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue(NoWarper::getName()));
  return res;
}

bool NoWarper::Factory::needsInputPreProcessing() const { return false; }

std::string NoWarper::Factory::hash() const {
  std::stringstream ss;
  ss << "NoImageWarper";
  return ss.str();
}

std::string NoWarper::Factory::getImageWarperName() const { return NoWarper::getName(); }

Potential<ImageWarper> NoWarper::Factory::create() const {
  return Potential<ImageWarper>(new NoWarper(std::map<std::string, float>()));
}

ImageWarperFactory* NoWarper::Factory::clone() const { return new Factory(); }

NoWarper::NoWarper(const std::map<std::string, float>& parameters) : ImageWarper(parameters) {}

std::string NoWarper::getName() { return std::string("no"); }

bool NoWarper::needImageFlow() const { return false; }

ImageWarper::ImageWarperAlgorithm NoWarper::getWarperAlgorithm() const {
  return ImageWarper::ImageWarperAlgorithm::NoWarper;
}

}  // namespace Core
}  // namespace VideoStitch
