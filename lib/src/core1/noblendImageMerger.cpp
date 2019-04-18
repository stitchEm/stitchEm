// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "noblendImageMerger.hpp"

#include "imageMapping.hpp"
#include "gpu/core1/mergerKernel.hpp"
#include "libvideostitch/ptv.hpp"
#include "parse/json.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageMerger> NoBlendImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                           const ImageMerger* to, bool) const {
  return Potential<ImageMerger>(new NoBlendImageMerger(pano, fromIm, to));
}

ImageMergerFactory* NoBlendImageMerger::Factory::clone() const { return new Factory(); }

Ptv::Value* Core::NoBlendImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("noblendv1"));
  return res;
}

NoBlendImageMerger::NoBlendImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm, const ImageMerger* to)
    : ImageMerger(fromIm.getImId(), to) {}

NoBlendImageMerger::~NoBlendImageMerger() {}

std::string NoBlendImageMerger::Factory::hash() const { return "v1_NoBlendImageMerger"; }

Status NoBlendImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                                      GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm, bool,
                                      GPU::Stream stream) const {
  return noblend(t, pano, pbo, fromIm, stream);
}

}  // namespace Core
}  // namespace VideoStitch
