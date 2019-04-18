// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/ptv.hpp"
#include "libvideostitch/audio.hpp"

#include "rtmpStructures.hpp"

#include <memory>
#include <vector>

#include <stdint.h>

namespace VideoStitch {
namespace Output {

class AudioEncoder {
 public:
  virtual ~AudioEncoder() {}

  static std::unique_ptr<AudioEncoder> createAudioEncoder(const Ptv::Value& config, const Audio::SamplingRate,
                                                          const Audio::SamplingDepth, const Audio::ChannelLayout,
                                                          const std::string& encoderType);

  virtual char* metadata(char* ptr, char* ptrEnd) = 0;
  virtual VideoStitch::IO::DataPacket* header() { return nullptr; }

  virtual int getBitRate() const = 0;

  virtual bool encode(mtime_t date, uint8_t* const* input, unsigned int numInputFrames,
                      std::vector<VideoStitch::IO::DataPacket>& packets) = 0;
};

}  // namespace Output
}  // namespace VideoStitch
