// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/buffer.hpp>
#include <gpu/stream.hpp>
#include "libvideostitch/status.hpp"
#include "maskMerger.hpp"

namespace VideoStitch {
namespace Core {

class VoronoiMaskMerger : public MaskMerger {
 public:
  VoronoiMaskMerger() : feather(0) {}
  ~VoronoiMaskMerger() {}

  Status setup(const PanoDefinition&, GPU::Buffer<const uint32_t> inputsMask, const ImageMapping& fromIm,
               const ImageMerger* const to, GPU::Stream) override;

  virtual Status updateAsync() override;

  virtual Status setParameters(const std::vector<double>& params) override;

 private:
  int feather;
  // with one parameter from 0 to 100, be able to provide different settings from sharp to smooth
  // - blending should not change when pano output size changes
  // - provide way to limit overlap
  static std::pair<float, float> transitionParameters(int feather);
};
}  // namespace Core
}  // namespace VideoStitch
