// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef AACDECODER_HPP
#define AACDECODER_HPP

#include "audioDecoder.hpp"

#if defined(_WIN32)
#include "faad.h"
#else
#include <neaacdec.h>
#endif

#include "libvideostitch/audio.hpp"

namespace VideoStitch {
namespace Input {

class AACDecoder : public AudioDecoder {
 public:
  AACDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth, const uint8_t nbChannels);
  ~AACDecoder();

  virtual void demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt);
  bool decode(VideoStitch::IO::DataPacket* pkt);
  virtual std::string name() { return "aac"; }

 private:
  NeAACDecHandle faad;

  bool initialised;
};

}  // namespace Input
}  // namespace VideoStitch

#endif  // AACDECODER_HPP
