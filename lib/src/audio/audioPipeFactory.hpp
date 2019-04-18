// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "audioPipeline.hpp"

#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/stitchOutput.hpp"

namespace VideoStitch {
namespace Audio {

using namespace Core;

class AudioPipeFactory {
 public:
  static Potential<AudioPipeline> create(const AudioPipeDefinition &audioPipeDef, const PanoDefinition &pano);
  ~AudioPipeFactory();

 private:
  AudioPipeFactory(AudioPipeDefinition &audioPipeDef);
};

}  // namespace Audio
}  // namespace VideoStitch
