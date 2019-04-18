// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./noFlow.hpp"

#include "parse/json.hpp"

#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Core {

NoFlow::Factory::Factory() {}

Potential<ImageFlowFactory> NoFlow::Factory::parse(const Ptv::Value&) {
  return Potential<ImageFlowFactory>(new NoFlow::Factory());
}

Ptv::Value* NoFlow::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue(NoFlow::getName()));
  return res;
}

std::string NoFlow::Factory::getImageFlowName() const { return NoFlow::getName(); }

bool NoFlow::Factory::needsInputPreProcessing() const { return false; }

std::string NoFlow::Factory::hash() const {
  std::stringstream ss;
  ss << "No Flow";
  return ss.str();
}

Potential<ImageFlow> NoFlow::Factory::create() const {
  std::map<std::string, float> parameters;
  return Potential<ImageFlow>(new NoFlow(parameters));
}

ImageFlowFactory* NoFlow::Factory::clone() const { return new Factory(); }

NoFlow::NoFlow(const std::map<std::string, float>& parameters) : ImageFlow(parameters) {}

std::string NoFlow::getName() { return std::string("no"); }

Status NoFlow::findSingleScaleImageFlow(const int2&, const int2&, const GPU::Buffer<const uint32_t>&, const int2&,
                                        const int2&, const GPU::Buffer<const uint32_t>&, GPU::Buffer<float2>,
                                        GPU::Stream) {
  // Do nothing
  return Status::OK();
}

Status NoFlow::upsampleFlow(const int2&, const int2&, const GPU::Buffer<const uint32_t>&, const int2&, const int2&,
                            const GPU::Buffer<const uint32_t>&, const GPU::Buffer<const float2>&, GPU::Buffer<float2>,
                            GPU::Stream) {
  // Do nothing
  return Status::OK();
}

ImageFlow::ImageFlowAlgorithm NoFlow::getFlowAlgorithm() const { return ImageFlow::ImageFlowAlgorithm::NoFlow; }

}  // namespace Core
}  // namespace VideoStitch
