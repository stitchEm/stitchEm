// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"

#include <thread>

namespace VideoStitch {
namespace Testing {

class MockVideoWriter : public Output::VideoWriter {
 public:
  MockVideoWriter(const std::string& n, unsigned w, unsigned h, int timeToPush, std::atomic<int>* numCalls,
                  VideoStitch::PixelFormat format = PixelFormat::RGBA)
      : Output(n), VideoWriter(w, h, {60, 1}, format), numCalls(numCalls), timeToPush(timeToPush) {
    *numCalls = 0;
  }

  void pushVideo(const Frame&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(timeToPush));
    ++*numCalls;
  }

 private:
  std::atomic<int>* numCalls;
  const int timeToPush;
};

class MockVideoWriter2 : public Output::VideoWriter {
 public:
  MockVideoWriter2(const std::string& n, unsigned w, unsigned h, VideoStitch::PixelFormat format)
      : Output(n), VideoWriter(w, h, {60, 1}, format), lastFrame(getExpectedFrameSize()) {}

  void pushVideo(const Frame& frame) {
    if (getPixelFormat() == RGBA || getPixelFormat() == RGB) {
      memcpy(lastFrame.data(), frame.planes[0], lastFrame.size());
    } else if (getPixelFormat() == YV12) {
      memcpy(lastFrame.data(), frame.planes[0], frame.width * frame.height);
      memcpy(lastFrame.data() + frame.width * frame.height, frame.planes[1], frame.width * frame.height / 4);
      memcpy(lastFrame.data() + frame.width * frame.height * 5 / 4, frame.planes[2], frame.width * frame.height / 4);
    } else {
      assert(false);
    }
  }

  const std::vector<unsigned char>& lastFrameData() const { return lastFrame; }

 private:
  std::vector<unsigned char> lastFrame;
};

}  // namespace Testing
}  // namespace VideoStitch
