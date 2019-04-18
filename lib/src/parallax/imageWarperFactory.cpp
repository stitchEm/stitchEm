// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/imageWarperFactory.hpp"

#include "parallax/noWarper.hpp"
#include "parallax/noFlow.hpp"
#ifndef VS_OPENCL
#include "parallax/linearFlowWarper.hpp"
#include "parallax/simpleFlow.hpp"
#endif

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <cassert>
#include <mutex>
#include <iostream>

namespace VideoStitch {
namespace Core {

Potential<Core::ImageWarperFactory> ImageWarperFactory::createWarperFactory(const Ptv::Value* value) {
  if (!value) {
    std::unique_ptr<Ptv::Value> emptyObject = std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
    return NoWarper::Factory::parse(*emptyObject.get());
  }
  // Make sure value is an object, if not, return nowarper
  if (!Parse::checkType("ImageWarperFactory", *value, Ptv::Value::OBJECT)) {
    return Potential<Core::ImageWarperFactory>(new NoWarper::Factory());
  }
  std::string type;
  if (Parse::populateString("ImageWarperFactory", *value, "type", type, false) == Parse::PopulateResult_WrongType) {
    return {Origin::ImageWarper, ErrType::InvalidConfiguration,
            "Invalid type for 'type' configuration, expected string"};
  }
  if (type == NoWarper::getName() || type == "") {
    return NoWarper::Factory::parse(*value);
#ifndef VS_OPENCL
  } else if (type == LinearFlowWarper::getName()) {
    return LinearFlowWarper::Factory::parse(*value);
#endif
  } else {
    Logger::get(Logger::Error) << "Unknown warper type: '" << type << "'." << std::endl;
    return {Origin::ImageWarper, ErrType::InvalidConfiguration, "Invalid warper type"};
  }
}

const std::vector<std::string>& ImageWarperFactory::availableWarpers() {
  static std::vector<std::string> availableWarpers;
  // Lazily fill in the list of mergers. TODO: use a better, macro-based, registration pattern.
  static std::mutex mutex;
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (availableWarpers.empty()) {
      availableWarpers.push_back(NoWarper::getName());
#ifndef VS_OPENCL
      availableWarpers.push_back(LinearFlowWarper::getName());
#endif
    }
  }
  return availableWarpers;
}

std::vector<std::string> ImageWarperFactory::compatibleWarpers(const std::string& flow) {
  std::vector<std::string> compatibleWarpers;
  // Lazily fill in the list of mergers. TODO: use a better, macro-based, registration pattern.
  static std::mutex mutex;
  {
    std::unique_lock<std::mutex> lock(mutex);
    compatibleWarpers.clear();
    if (flow == NoFlow::getName()) {
      compatibleWarpers.push_back(NoWarper::getName());
    }
#ifndef VS_OPENCL
    if (flow == SimpleFlow::getName()) {
      compatibleWarpers.push_back(LinearFlowWarper::getName());
    }
#endif
  }
  return compatibleWarpers;
}

bool ImageWarperFactory::equal(const ImageWarperFactory& other) const { return hash() == other.hash(); }

namespace {
/**
 * A merger factory that cannot instantiate a merger. This can be useful when creating a merger without stitchers.
 */
class ImpotentWarperFactory : public ImageWarperFactory {
 public:
  virtual Potential<ImageWarper> create() const override {
    return {Origin::ImageWarper, ErrType::ImplementationError, "ImpotentWarper is not implemented properly"};
  }

  virtual ~ImpotentWarperFactory() {}

  virtual Ptv::Value* serialize() const override { return nullptr; }

  virtual std::string getImageWarperName() const override { return "ImpotentWarper"; }

  virtual bool needsInputPreProcessing() const override { return false; }

  virtual ImpotentWarperFactory* clone() const override { return new ImpotentWarperFactory(); }

  virtual std::string hash() const override { return "ImpotentWarper"; }
};
}  // namespace

Potential<ImageWarperFactory> ImageWarperFactory::newImpotentWarperFactory() {
  return Potential<ImageWarperFactory>(new ImpotentWarperFactory);
}

}  // namespace Core
}  // namespace VideoStitch
