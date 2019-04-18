// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioEncoder.hpp"
#include "aacEncoder.hpp"
#include "lameEncoder.hpp"

namespace VideoStitch {
namespace Output {

std::unique_ptr<AudioEncoder> AudioEncoder::createAudioEncoder(const Ptv::Value& config,
                                                               const Audio::SamplingRate samplingRate,
                                                               const Audio::SamplingDepth samplingDepth,
                                                               const Audio::ChannelLayout channelLayout,
                                                               const std::string& encoderType) {
  if ("mp3" == encoderType) {
    return MP3Encoder::createMP3Encoder(config, samplingRate, samplingDepth, channelLayout);
  } else if ("aac" == encoderType) {
    return AACEncoder::createAACEncoder(config, samplingRate, samplingDepth, channelLayout);
  }

  return std::unique_ptr<AudioEncoder>();
}

}  // namespace Output
}  // namespace VideoStitch
