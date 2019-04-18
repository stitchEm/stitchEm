// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LAMEDECODER_HPP
#define LAMEDECODER_HPP

#include "audioDecoder.hpp"
#include "amfIncludes.hpp"

#include "libvideostitch/audio.hpp"

#include "lame/lame.h"

namespace VideoStitch {
namespace Input {

class MP3Decoder : public AudioDecoder {
 public:
  MP3Decoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth, const uint8_t nbChannels);
  ~MP3Decoder();

  virtual void demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt);
  bool decode(VideoStitch::IO::DataPacket* pkt);
  virtual std::string name() { return "mp3"; }

 private:
  hip_t hipLame;

  std::vector<short> pcm_l;
  std::vector<short> pcm_r;
};

}  // namespace Input
}  // namespace VideoStitch

#endif  // LAMEDECODER_HPP
