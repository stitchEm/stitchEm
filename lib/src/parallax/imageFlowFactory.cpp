// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/imageFlowFactory.hpp"

#include "parallax/noFlow.hpp"
#ifndef VS_OPENCL
#include "parallax/simpleFlow.hpp"
#endif  // VS_OPENCL

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <cassert>
#include <mutex>
#include <iostream>

namespace VideoStitch {
namespace Core {

Potential<Core::ImageFlowFactory> Core::ImageFlowFactory::createFlowFactory(const Ptv::Value* value) {
  // Make sure value is an object.
  if (!value) {
    std::unique_ptr<Ptv::Value> emptyObject = std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
    return NoFlow::Factory::parse(*emptyObject.get());
  }

  if (!Parse::checkType("ImageFlowFactory", *value, Ptv::Value::OBJECT)) {
    return Potential<Core::ImageFlowFactory>(new NoFlow::Factory());
  }
  std::string type;
  if (Parse::populateString("ImageFlowFactory", *value, "type", type, false) == Parse::PopulateResult_WrongType) {
    return {Origin::ImageFlow, ErrType::InvalidConfiguration, "Invalid type for 'type' configuration, expected string"};
  }
  if (type == NoFlow::getName() || type == "") {
    return NoFlow::Factory::parse(*value);
#ifndef VS_OPENCL
  } else if (type == SimpleFlow::getName()) {
    return SimpleFlow::Factory::parse(*value);
#endif  // VS_OPENCL
  } else {
    Logger::get(Logger::Error) << "Unknown flow type: '" << type << "'." << std::endl;
    return {Origin::ImageFlow, ErrType::InvalidConfiguration, "Invalid warper type"};
  }
}

const std::vector<std::string>& Core::ImageFlowFactory::availableFlows() {
  static std::vector<std::string> availableFlows;
  // Lazily fill in the list of mergers. TODO: use a better, macro-based, registration pattern.
  static std::mutex mutex;
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (availableFlows.empty()) {
      availableFlows.push_back(NoFlow::getName());
#ifndef VS_OPENCL
      availableFlows.push_back(SimpleFlow::getName());
#endif
    }
  }
  return availableFlows;
}

bool ImageFlowFactory::equal(const ImageFlowFactory& other) const { return hash() == other.hash(); }

namespace {
/**
 * A merger factory that cannot instantiate a merger. This can be useful when creating a merger without stitchers.
 */
class ImpotentFlowFactory : public ImageFlowFactory {
 public:
  virtual Potential<ImageFlow> create() const override {
    return {Origin::ImageFlow, ErrType::ImplementationError, "ImpotentFlow is not implemented properly"};
  }
  virtual ~ImpotentFlowFactory() {}

  virtual Ptv::Value* serialize() const override { return nullptr; }

  virtual bool needsInputPreProcessing() const override { return false; }

  virtual std::string getImageFlowName() const override { return "ImpotentFlow"; }

  virtual ImpotentFlowFactory* clone() const override { return new ImpotentFlowFactory(); }
  virtual std::string hash() const override { return "ImpotentFlow"; }
};
}  // namespace

Potential<ImageFlowFactory> ImageFlowFactory::newImpotentFlowFactory() {
  return Potential<ImageFlowFactory>(new ImpotentFlowFactory);
}

}  // namespace Core
}  // namespace VideoStitch
