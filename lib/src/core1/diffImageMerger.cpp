// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "diffImageMerger.hpp"

#include "imageMapping.hpp"

#include "gpu/core1/mergerKernel.hpp"
#include "parse/json.hpp"

#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageMerger> DiffImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                        const ImageMerger* to, bool) const {
  return Potential<ImageMerger>(new DiffImageMerger(pano, fromIm, to));
}

ImageMergerFactory* DiffImageMerger::Factory::clone() const { return new Factory(); }

Ptv::Value* Core::DiffImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("diff"));
  return res;
}

DiffImageMerger::DiffImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm, const ImageMerger* to)
    : ImageMerger(fromIm.getImId(), to) {}

DiffImageMerger::~DiffImageMerger() {}

std::string DiffImageMerger::Factory::hash() const { return "v1_DiffImageMerger"; }

Status DiffImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                                   GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm, bool,
                                   GPU::Stream stream) const {
  return stitchingError(t, pano, pbo, fromIm, stream);
}

Status DiffImageMerger::reconstruct(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut, bool,
                                    GPU::Stream stream) const {
  return amplitude(pano, panoDevOut, stream);
}
}  // namespace Core
}  // namespace VideoStitch
