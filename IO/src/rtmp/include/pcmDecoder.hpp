// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PCMDECODER_HPP
#define PCMDECODER_HPP

#include "audioDecoder.hpp"

namespace VideoStitch {
namespace Input {

class PCMDecoder : public AudioDecoder {
 public:
  PCMDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth, const uint8_t nbChannels);

  virtual void demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt);
  virtual bool decode(VideoStitch::IO::DataPacket* pkt);
  virtual std::string name();

 private:
  bool warn;
};

}  // namespace Input
}  // namespace VideoStitch

#endif  // PCMDECODER_HPP
