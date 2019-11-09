// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "rtmpStructures.hpp"
#include "audioEncoder.hpp"
#include "ptvMacro.hpp"
#include "librtmpIncludes.hpp"

#if defined(_WIN32)
#include "faac.h"
#else
#include <faac.h>
#endif

#include "libvideostitch/audio.hpp"

namespace VideoStitch {
namespace Output {

class AACEncoder : public AudioEncoder {
 public:
  ~AACEncoder() { faacEncClose(faac); }

  static std::unique_ptr<AudioEncoder> createAACEncoder(const Ptv::Value& config, const Audio::SamplingRate rate,
                                                        const Audio::SamplingDepth depth,
                                                        const Audio::ChannelLayout layout);

  char* metadata(char* enc, char* pend);

  VideoStitch::IO::DataPacket* header() { return &headerPkt; }

  bool encode(mtime_t date, uint8_t* const* input, unsigned int numInputFrames,
              std::vector<VideoStitch::IO::DataPacket>& packets);

  int getBitRate() const { return bitRate; }

 private:
  AACEncoder(unsigned int bitRate, int sampleRate, int nbChans, Audio::SamplingDepth fmt,
             const std::vector<int64_t>& channelMap);

  static const int DEFAULT_AUDIO_BITRATE;

  static const AVal av_audiocodecid;
  static const AVal av_mp4a;
  static const AVal av_audiodatarate;
  static const AVal av_audiosamplerate;
  static const AVal av_audiosamplesize;
  static const AVal av_audiochannels;
  static const AVal av_stereo;

  faacEncHandle faac;
  unsigned long inputSamples;  // the total number of samples that should be fed to faacEncEncode() in each call

  unsigned int bitRate;
  int sampleRate;
  int nbChans;
  int sampleDepth;
  Audio::SamplingDepth sampleFormat;

  std::vector<float> inputBuffer;
  void stuffInputBuffer(uint8_t* input, int numSamples);
  int framesFed = 0;  // frames currently in the encoder

  // AAC raw
  std::vector<unsigned char> aacbuf;
  // AAC sequence header
  VideoStitch::IO::DataPacket headerPkt;
};

}  // namespace Output
}  // namespace VideoStitch
