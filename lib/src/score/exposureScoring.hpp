// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"

namespace VideoStitch {
namespace Scoring {

class VS_EXPORT ExposureScoringAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;

 public:
  explicit ExposureScoringAlgorithm(const Ptv::Value* config);
  virtual ~ExposureScoringAlgorithm();
  Potential<Ptv::Value> apply(Core::PanoDefinition*, ProgressReporter*, Util::OpaquePtr** ctx = nullptr) const override;

 private:
  int firstFrame;
  int lastFrame;
};

}  // namespace Scoring
}  // namespace VideoStitch
