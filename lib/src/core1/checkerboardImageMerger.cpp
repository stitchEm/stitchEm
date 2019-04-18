// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "checkerboardImageMerger.hpp"

#include "imageMapping.hpp"

#include "gpu/core1/mergerKernel.hpp"
#include "parse/json.hpp"

#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Core {

CheckerboardImageMerger::Factory::Factory(int checkerSize) : checkerSize(checkerSize) {}

Potential<ImageMerger> CheckerboardImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                                const ImageMerger* to, bool) const {
  return new CheckerboardImageMerger(pano, fromIm, to, checkerSize);
}

Potential<ImageMergerFactory> CheckerboardImageMerger::Factory::parse(const Ptv::Value& value) {
  int checkerSize = 16;
  // hijack the "feather" setting to be able to control the checker size through the UI
  if (Parse::populateInt("CheckerboardImageMergerFactory", value, "feather", checkerSize, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'feather' in checkerboard merger configuration, expected integer value"};
  }
  checkerSize = std::max(checkerSize, 0);
  return Potential<ImageMergerFactory>(new CheckerboardImageMerger::Factory(checkerSize));
}

ImageMergerFactory* CheckerboardImageMerger::Factory::clone() const { return new Factory(checkerSize); }

Ptv::Value* Core::CheckerboardImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("checkerboard"));
  res->push("feather", new Parse::JsonValue((int)checkerSize));
  return res;
}

CheckerboardImageMerger::CheckerboardImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm,
                                                 const ImageMerger* to, int checkerSize)
    : ImageMerger(fromIm.getImId(), to), checkerSize(checkerSize) {}

CheckerboardImageMerger::~CheckerboardImageMerger() {}

std::string CheckerboardImageMerger::Factory::hash() const {
  std::stringstream ss;
  ss << "v1_CheckerboardImageMerger" << checkerSize;
  return ss.str();
}

Status CheckerboardImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano,
                                           GPU::Buffer<uint32_t> panoDevOut, GPU::UniqueBuffer<uint32_t>&,
                                           const ImageMapping& fromIm, bool, GPU::Stream stream) const {
  return checkerMerge(t, pano, panoDevOut, fromIm, checkerSize, stream);
}

}  // namespace Core
}  // namespace VideoStitch
