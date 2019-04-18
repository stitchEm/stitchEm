// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/postprocessor.hpp"

namespace VideoStitch {
namespace Scoring {

class ScoringPostProcessor : public Core::PostProcessor {
 public:
  static ScoringPostProcessor* create();
  Status process(GPU::Buffer<uint32_t>& devBuffer, const Core::PanoDefinition& pano, frameid_t frame,
                 GPU::Stream& stream) const;
  void getScore(double& normalized_cross_correlation, double& uncovered_ratio) const;

 private:
  ScoringPostProcessor() : m_score(0.), m_uncovered(0.){};
  mutable double m_score;
  mutable double m_uncovered;
};

}  // namespace Scoring
}  // namespace VideoStitch
