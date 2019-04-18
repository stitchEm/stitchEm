// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imageMerger.hpp"

#include "imageMapping.hpp"

#include "libvideostitch/profile.hpp"

namespace VideoStitch {
namespace Core {

ImageMerger::ImageMerger(videoreaderid_t imId, const ImageMerger* to)
    : to(to), idMask(1 << imId | (to ? to->getIdMask() : 0)) {}

ImageMerger::~ImageMerger() {}

Status ImageMerger::reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t>, bool, GPU::Stream) const {
  return Status::OK();
}

bool ImageMerger::isMultiScale() const { return false; }

}  // namespace Core
}  // namespace VideoStitch
