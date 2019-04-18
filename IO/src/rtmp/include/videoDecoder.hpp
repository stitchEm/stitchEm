// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iopacket.hpp"
#include "rtmpStructures.hpp"

#include "libvideostitch/frame.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/span.hpp"

namespace VideoStitch {
namespace Input {

typedef void* VideoPtr;

#define CIRCULAR_BUFFER_LEN 100

class VideoDecoder {
 public:
  enum class Type { QuickSync, CuVid, Mock };

  static Potential<VideoDecoder> createVideoDecoder(int width, int height, FrameRate framerate, Type encoderType);

  // If 'dec' is a known decoder string, fill type and return true
  static PotentialValue<Type> parseDecoderType(const std::string& dec);

  static AddressSpace decoderAddressSpace(Type type);

  static std::string typeToString(Type decoderType);

  virtual ~VideoDecoder() {}

  static bool demuxPacket(Span<const unsigned char>, mtime_t, VideoStitch::IO::Packet&, std::vector<unsigned char>&);
  static void demuxHeader(Span<const unsigned char>, mtime_t, VideoStitch::IO::Packet&, std::vector<unsigned char>&);

  virtual bool demux(Span<const unsigned char>, mtime_t, VideoStitch::IO::Packet&) { return true; }
  virtual void decodeHeader(Span<const unsigned char> pkt, mtime_t timestamp, Span<unsigned char>&) = 0;
  virtual bool decodeAsync(VideoStitch::IO::Packet&) = 0;
  virtual bool synchronize(mtime_t& timestamp, VideoPtr& pic) = 0;

  virtual void copyFrame(unsigned char*, mtime_t&, VideoPtr) = 0;
  virtual void releaseFrame(VideoPtr) = 0;
  virtual size_t flush() = 0;

  virtual void stop() = 0;

  virtual std::string name() { return "libx264"; }
};

}  // namespace Input
}  // namespace VideoStitch
