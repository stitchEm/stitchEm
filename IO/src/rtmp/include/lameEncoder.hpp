// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "librtmpIncludes.hpp"
#include "audioEncoder.hpp"
#include "ptvMacro.hpp"

#include "libvideostitch/audio.hpp"

#include "lame/lame.h"

namespace VideoStitch {
namespace Output {

class MP3Encoder : public AudioEncoder {
 public:
  MP3Encoder(unsigned int bitRate, Audio::SamplingRate sampleRate, int nbChans, Audio::SamplingDepth fmt);

  ~MP3Encoder() { lame_close(lgf); }

  static std::unique_ptr<AudioEncoder> createMP3Encoder(const Ptv::Value& config,
                                                        const Audio::SamplingRate samplingRate,
                                                        Audio::SamplingDepth depth, const Audio::ChannelLayout layout);

  char* metadata(char* enc, char* pend);

  bool encode(mtime_t date, uint8_t* const* input, unsigned int numInputFrames,
              std::vector<VideoStitch::IO::DataPacket>& packets);

  int getBitRate() const { return bitRate; }

 private:
  static const int DEFAULT_AUDIO_BITRATE;
  static const int audioBlockSize;
  static const int frameSize;

  static const AVal av_audiocodecid;
  static const AVal av_audiodatarate;
  static const AVal av_audiosamplerate;
  static const AVal av_audiosamplesize;
  static const AVal av_audiochannels;
  static const AVal av_stereo;

  lame_global_flags* lgf;

  unsigned int bitRate;
  int sampleRate;
  int nbChans;
  Audio::SamplingDepth fmt;

  std::vector<unsigned char> mp3buf;
  std::vector<unsigned char> header;
};

}  // namespace Output
}  // namespace VideoStitch
