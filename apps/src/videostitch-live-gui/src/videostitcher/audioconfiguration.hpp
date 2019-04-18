// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audio.hpp"
#include "libvideostitch/ptv.hpp"

struct AudioConfiguration {
  AudioConfiguration() = default;
  explicit AudioConfiguration(const VideoStitch::Ptv::Value& readerConfig);

  bool isValid() const;
  bool isAja() const;
  VideoStitch::Ptv::Value* serialize() const;  // Gives ownership

  std::string type;
  std::string inputName;
  int nbAudioChannels = 0;
  VideoStitch::Audio::SamplingRate audioRate = VideoStitch::Audio::SamplingRate::SR_NONE;
};
