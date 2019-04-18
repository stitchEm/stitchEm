// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"

namespace VideoStitch {
namespace Scoring {

class ScoringPostProcessor;

class VS_EXPORT ScoringAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;

 public:
  explicit ScoringAlgorithm(const Ptv::Value* config);
  virtual ~ScoringAlgorithm();
  Potential<Ptv::Value> apply(Core::PanoDefinition*, ProgressReporter*, Util::OpaquePtr** ctx = NULL) const override;

 private:
  int firstFrame;
  int lastFrame;
};

}  // namespace Scoring
}  // namespace VideoStitch
