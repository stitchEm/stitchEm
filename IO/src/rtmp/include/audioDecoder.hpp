// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef AUDIODECODER_H
#define AUDIODECODER_H

#include "iopacket.hpp"
#include "rtmpStructures.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/circularBuffer.hpp"

#include "librtmpIncludes.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace VideoStitch {
namespace Input {

struct AudioStream {
  CircularBuffer<uint8_t> stream;
  cntime_t cnts;
  std::mutex audioBufferMutex;

  AudioStream() : stream(2048), cnts(0) {}
};

class AudioDecoder {
 public:
  AudioDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth,
               const uint8_t nbChannels);
  virtual ~AudioDecoder() = default;

  virtual void demux(VideoStitch::IO::DataPacket&, VideoStitch::IO::Packet&) {}
  virtual bool decode(VideoStitch::IO::DataPacket* pkt) = 0;
  virtual std::string name() = 0;

  static std::unique_ptr<AudioDecoder> createAudioDecoder(AMFDataType encoderType, AMFObjectProperty* amfOProperty,
                                                          AudioStream* audioStream, const long samplingRate,
                                                          const int samplingDepth, const int nbChannels);

 protected:
  uint64_t samplingRate;
  uint8_t samplingDepth;
  uint8_t numberOfChannels;
  AudioStream* audioStream;
};

}  // namespace Input
}  // namespace VideoStitch
#endif  // AUDIODECODER_H
