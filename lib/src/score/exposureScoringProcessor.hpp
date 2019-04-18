// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/postprocessor.hpp"

#include <array>

namespace VideoStitch {
namespace Scoring {

class ExposureScoringPostProcessor : public Core::PostProcessor {
 public:
  static ExposureScoringPostProcessor* create();
  Status process(GPU::Buffer<uint32_t>& devBuffer, const Core::PanoDefinition& pano, frameid_t frame,
                 GPU::Stream& stream) const;

  std::array<double, 3> getScore() const { return rgbDiff; };

 private:
  ExposureScoringPostProcessor(){};
  mutable std::array<double, 3> rgbDiff;
};

}  // namespace Scoring
}  // namespace VideoStitch
