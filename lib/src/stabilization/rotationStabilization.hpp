// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"

#include "motion/rotationalMotion.hpp"

#include <map>

namespace VideoStitch {
namespace Stab {

/**
 * @brief Algorithm aiming at reducing annoying jitter in panoramic
 * video.
 * This one aims at smoothing only the orientation of the point-of-view,
 * in terms of yaw-pitch-roll.
 *
 * See:
 *   https://www.cvl.isy.liu.se/education/graduate/geometry2010/lectures/Lecture7b.pdf
 *
 */
class VS_EXPORT RotationStabilizationAlgorithm : public Util::Algorithm {
 public:
  typedef Motion::RotationalMotionModelEstimation::MotionModel MotionModel;

  class StabContext : public Util::OpaquePtr {
   public:
    MotionModel models;
  };

  static const char* docString;

  explicit RotationStabilizationAlgorithm(const Ptv::Value* config);
  virtual ~RotationStabilizationAlgorithm();

  Potential<Ptv::Value> apply(Core::PanoDefinition*, ProgressReporter*, Util::OpaquePtr** ctx) const override;

 private:
  int64_t firstFrame;
  mutable int64_t lastFrame;
  int64_t convolutionSpan;
  std::vector<int> devices;
};

}  // namespace Stab
}  // namespace VideoStitch
