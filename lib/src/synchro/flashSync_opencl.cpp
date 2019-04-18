// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "flashSync.hpp"

#include "util/registeredAlgo.hpp"

namespace VideoStitch {
namespace Synchro {
namespace {
Util::RegisteredAlgo<FlashSyncAlgorithm> registered("flash_synchronization");
}
const char* FlashSyncAlgorithm::docString =
    "An algorithm that computes frame offsets using the luma histograms to synchronize the inputs.\n"
    "Can be applied pre-calibration.\n"
    "The result is a { \"frames\": list of integer offsets (all >=0, in frames), \"seconds\": list of double offsets "
    "(all >=0.0, in seconds) }\n"
    "which can be used directly as a 'frame_offset' parameter for the 'inputs'.\n";

FlashSyncAlgorithm::FlashSyncAlgorithm(const Ptv::Value* /*config*/) : firstFrame(0), lastFrame(0) {
  (void)firstFrame;
  (void)lastFrame;
}

FlashSyncAlgorithm::~FlashSyncAlgorithm() {}

Potential<Ptv::Value> FlashSyncAlgorithm::apply(Core::PanoDefinition* /*pano*/, ProgressReporter* /*progress*/,
                                                Util::OpaquePtr**) const {
  return Potential<Ptv::Value>(Origin::SynchronizationAlgorithm, ErrType::UnsupportedAction,
                               "Flash synch not supported");
}

Status FlashSyncAlgorithm::doAlign(const std::vector<int>& /*devices*/, const Core::PanoDefinition& /*pano*/,
                                   std::vector<int>& /*frames*/, ProgressReporter* /*progress*/) const {
  return {Origin::SynchronizationAlgorithm, ErrType::UnsupportedAction, "Flash synch not supported"};
}
}  // namespace Synchro
}  // namespace VideoStitch
