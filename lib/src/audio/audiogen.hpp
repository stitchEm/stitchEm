// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audio.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Audio {
/**
 * A factory for audio readers that use a generator to create a synthetic audio input.
 */
class AudioGenFactory {
 public:
  static Input::AudioReader* create(readerid_t id, const Ptv::Value& config);
};
}  // namespace Audio
}  // namespace VideoStitch
