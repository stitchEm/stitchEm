// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rtmpClient.hpp"

namespace VideoStitch {
namespace Input {

class MockVideoDecoder : public VideoDecoder {
 public:
  virtual void decodeHeader(Span<const unsigned char>, mtime_t, Span<unsigned char>&) override {}

  virtual bool decodeAsync(VideoStitch::IO::Packet&) override { return true; }

  virtual bool synchronize(mtime_t&, VideoPtr&) override { return true; }

  virtual void copyFrame(unsigned char*, mtime_t&, VideoPtr) override {}

  virtual void releaseFrame(VideoPtr) override {}

  virtual size_t flush() override { return 0; }

  void stop() override {}
};

VideoDecoder* createMockDecoder(int /* width */, int /* height */, FrameRate /* framerate */) {
  return new MockVideoDecoder();
}

}  // namespace Input
}  // namespace VideoStitch
