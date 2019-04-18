// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureDiffImageMerger.hpp"

#include "imageMapping.hpp"

#include "gpu/core1/mergerKernel.hpp"
#include "parse/json.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageMerger> ExposureDiffImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                                const ImageMerger* to, bool) const {
  return new ExposureDiffImageMerger(pano, fromIm, to);
}

ImageMergerFactory* ExposureDiffImageMerger::Factory::clone() const { return new Factory(); }

Ptv::Value* Core::ExposureDiffImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("exposure_diff"));
  return res;
}

ExposureDiffImageMerger::ExposureDiffImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm,
                                                 const ImageMerger* to)
    : ImageMerger(fromIm.getImId(), to) {}

ExposureDiffImageMerger::~ExposureDiffImageMerger() {}

std::string ExposureDiffImageMerger::Factory::hash() const { return "v1_ExposureDiffImageMerger"; }

Status ExposureDiffImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano,
                                           GPU::Buffer<uint32_t> panoDevOut, GPU::UniqueBuffer<uint32_t>&,
                                           const ImageMapping& fromIm, bool, GPU::Stream stream) const {
  return exposureDiffRGB(t, pano, panoDevOut, fromIm, stream);
}

Status ExposureDiffImageMerger::reconstruct(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                                            bool, GPU::Stream stream) const {
  return disregardNoDiffArea(pano, panoDevOut, stream);
}

}  // namespace Core
}  // namespace VideoStitch
