// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Synchro {

/**
 * @brief Detection of synchronization frame offset using luma histograms.
 *
 * Works only if a flash occured somewhere in the sequence.
 *
 */
class VS_EXPORT FlashSyncAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;

  explicit FlashSyncAlgorithm(const Ptv::Value* config);
  virtual ~FlashSyncAlgorithm();

  Potential<Ptv::Value> apply(Core::PanoDefinition*, ProgressReporter*, Util::OpaquePtr** ctx = NULL) const override;

 private:
  /**
   * Do the actual work. If @a seconds is not NULL, also populate the offsets in seconds.
   */
  virtual Status doAlign(const std::vector<int>& devices, const Core::PanoDefinition&, std::vector<int>& frames,
                         ProgressReporter*) const;

  frameid_t firstFrame;
  frameid_t lastFrame;
  std::vector<int> devices;
};

}  // namespace Synchro
}  // namespace VideoStitch
