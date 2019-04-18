// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mergerMaskAlgorithm.hpp"

#include "util/registeredAlgo.hpp"

namespace VideoStitch {
namespace MergerMask {

namespace {
Util::RegisteredAlgo<MergerMaskAlgorithm> registered("mask");
}

const char* MergerMaskAlgorithm::docString =
    "An algorithm that optimizes the blending mask and blending order of the input images\n";

MergerMaskAlgorithm::MergerMaskAlgorithm(const Ptv::Value* config) : mergerMaskConfig(config), rigDef(nullptr) {
  (void)rigDef;
}

MergerMaskAlgorithm::~MergerMaskAlgorithm() {}

Potential<Ptv::Value> MergerMaskAlgorithm::apply(Core::PanoDefinition* /*pano*/, ProgressReporter* /*progress*/,
                                                 Util::OpaquePtr**) const {
  return Potential<Ptv::Value>(Origin::MaskInterpolationAlgorithm, ErrType::UnsupportedAction,
                               "Merger mask algorithm not supported");
}

}  // namespace MergerMask
}  // namespace VideoStitch
