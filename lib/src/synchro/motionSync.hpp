// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "motion/affineMotion.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Synchro {

/**
 * @brief Detection of synchronization frame offset using global motion model estimates.
 *
 * See:
 *   http://www.crcv.ucf.edu/papers/Spencer_Shah_accv_2004.pdf
 */
class VS_EXPORT MotionSyncAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;

  explicit MotionSyncAlgorithm(const Ptv::Value* config);
  virtual ~MotionSyncAlgorithm();

  Potential<Ptv::Value> apply(Core::PanoDefinition*, ProgressReporter*, Util::OpaquePtr** ctx = NULL) const override;

 private:
  Status doAlignUsingFarneback(const Core::PanoDefinition&, std::vector<int>& frames, std::vector<double>& costs,
                               bool& success, ProgressReporter*) const;

  Status computeFeaturesForNCC(std::size_t nbInputs, std::size_t nbFrames,
                               const std::vector<std::vector<double> >& magnitudes,
                               const std::vector<Motion::AffineMotionModelEstimation::MotionModel>& motionModels,
                               std::vector<std::vector<std::vector<double> > >& features,
                               std::vector<double>& featureWeights) const;

  static Status alignGivenFeatures(const std::vector<std::vector<std::vector<double> > >& features,
                                   const std::vector<double>& featureWeights, const Core::PanoDefinition& pano,
                                   std::vector<int>& frames, std::vector<double>& costs, bool& success);

  inline bool useSpencerShahFeatures() const { return (rollWeight > 0) || (translationWeight > 0); }

  inline bool useFlowMedianMagnitudeDifferencesFeatures() const { return flowMedianMagnitudeDifferenceWeight > 0; }

  uint64_t firstFrame;
  mutable uint64_t lastFrame;
  std::vector<int> devices;
  const double rollWeight;
  const double translationWeight;
  const double flowMedianMagnitudeDifferenceWeight;
};

}  // namespace Synchro
}  // namespace VideoStitch
