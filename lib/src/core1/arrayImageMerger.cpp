// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "arrayImageMerger.hpp"

#include "imageMapping.hpp"

#include "gpu/core1/mergerKernel.hpp"
#include "parse/json.hpp"

#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageMerger> ArrayImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                         const ImageMerger* to, bool) const {
  return Potential<ImageMerger>(new ArrayImageMerger(pano, fromIm, to));
}

ImageMergerFactory* ArrayImageMerger::Factory::clone() const { return new Factory(); }

ArrayImageMerger::ArrayImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm, const ImageMerger* to)
    : ImageMerger(fromIm.getImId(), to) {}

ArrayImageMerger::~ArrayImageMerger() {}

std::string ArrayImageMerger::Factory::hash() const { return "v1_ArrayImageMerger"; }

Ptv::Value* Core::ArrayImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("array"));
  return res;
}

Status ArrayImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                                    GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm, bool,
                                    GPU::Stream stream) const {
  return countInputs(t, pano, panoDevOut, fromIm, stream);
}

Status ArrayImageMerger::reconstruct(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, bool,
                                     GPU::Stream stream) const {
  return colorMap(pano, pbo, stream);
}
}  // namespace Core
}  // namespace VideoStitch
