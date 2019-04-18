// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include <xiApi.h>
#include <queue>

namespace VideoStitch {
namespace Input {

typedef std::pair<mtime_t, unsigned char*> XFrame;

class DualBuffer {
 private:
  XFrame buff[2];
  uint8_t readPtr;
  uint8_t writePtr;

  uint64_t size;
  bool empty;
  bool full;
  bool stopLoop;
  std::mutex mtx;
  std::condition_variable rCond;
  std::condition_variable wCond;

 public:
  DualBuffer(int64_t widthParam, int64_t heightParam);
  ~DualBuffer();

  bool read(XFrame* data);
  bool write(XFrame* data);
  void stop() {
    stopLoop = true;
    wCond.notify_one();
  };
};

class XimeaReader : public VideoReader {
 public:
  static XimeaReader* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);

  ~XimeaReader();

  // ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) override;
  ReadStatus readFrame(mtime_t& date, unsigned char* video) override;
  Status seekFrame(frameid_t) override;
  // Status seekFrame(mtime_t) override;
  // size_t available() override;
  // bool eos() override;

  int deviceIndex() { return devIdx; };
  uint64_t bw() { return bandwidth; };
  uint64_t fpsLim() { return fpsLimit; }
  bool stopThread() { return stop; };

  // std::queue<Frame> imgQueue;

  DualBuffer* dBuff;

 private:
  XimeaReader(readerid_t id, const int64_t width, const int64_t height, int deviceIndex, const bool withAudio,
              FrameRate fps, bool interlaced, int bw, int frameRateLimit);

  static void ximeaThread(XimeaReader* XR);

  int devIdx;
  uint64_t bandwidth;
  uint64_t fpsLimit;

  bool stop;
  std::thread* thr;
};

}  // namespace Input
}  // namespace VideoStitch
