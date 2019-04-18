// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stackImageMerger.hpp"

#include "imageMapping.hpp"

#include "gpu/image/imgInsert.hpp"
#include "parse/json.hpp"

#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Core {

Potential<ImageMerger> StackImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                         const ImageMerger* to, bool) const {
  return Potential<ImageMerger>(new StackImageMerger(pano, fromIm, to));
}

ImageMergerFactory* StackImageMerger::Factory::clone() const { return new Factory(); }

std::string StackImageMerger::Factory::hash() const { return "v1_StackImageMerger"; }

Ptv::Value* Core::StackImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("array"));
  return res;
}

Status StackImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                                    GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm, bool,
                                    GPU::Stream stream) const {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }
  // insert into pano image
  return Image::imgInsertInto(
      pbo, pano.getWidth(), pano.getHeight(), fromIm.getDeviceOutputBuffer(t), fromIm.getOutputRect(t).getWidth(),
      fromIm.getOutputRect(t).getHeight(), fromIm.getOutputRect(t).left(), fromIm.getOutputRect(t).top(),
      GPU::Buffer<const unsigned char>(), fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth(), false, stream);
}

StackImageMerger::StackImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm, const ImageMerger* to)
    : ImageMerger(fromIm.getImId(), to) {}

}  // namespace Core
}  // namespace VideoStitch
