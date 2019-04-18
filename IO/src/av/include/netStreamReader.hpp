// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libavReader.hpp"

#include <atomic>
#include <future>
#include <condition_variable>
#include <mutex>
#include <queue>

struct AVFormatContext;
struct AVCodec;
struct AVFrame;
struct AVPacket;

namespace VideoStitch {
namespace Input {

/* Network Streaming Client for Vahana Input Plugin */

class netStreamReader : public LibavReader {
 public:
  // TODOLATERSTATUS replace by Input::ReadStatus
  enum NetStreamReadStatus { Ok, Error, Continue, EOS };

  static bool handles(const std::string& filename);

  netStreamReader(readerid_t id, const std::string& displayName, const int64_t width, const int64_t height,
                  const int firstFrame, const AVPixelFormat fmt, AddressSpace addrSpace,
                  struct AVFormatContext* formatCtx,
#ifdef QUICKSYNC
                  class QSVContext* qsvCtx,
#endif
                  struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
                  struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* vFRame,
                  struct AVFrame* audioFrame, Util::TimeoutHandler* interruptCallback, const signed videoIdx,
                  const signed audioIdx, const Audio::ChannelLayout layout, const Audio::SamplingRate samplingRate,
                  const Audio::SamplingDepth samplingDepth);
  virtual ~netStreamReader();

  ReadStatus readFrame(mtime_t& date, unsigned char* video) override;
  ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) override;

 private:
  void readNetPackets();
  void decodeVideo();
  void decodeAudio();

  std::thread handlePackets;
  std::thread handleVideo;
  std::thread handleAudio;

  std::mutex videoQueueMutex, audioQueueMutex;
  std::condition_variable cvDecodeVideo, cvDecodeAudio;
  std::queue<AVPacket*> videoPacketQueue, audioPacketQueue;
  std::atomic<bool> stoppingQueues;

  std::mutex videoFrameMutex;
  std::mutex audioBufferMutex;
  std::condition_variable cvNewFrame;
  std::condition_variable cvFrameConsumed;
  std::vector<unsigned char> frame;
  std::atomic<bool> frameAvailable;
  bool stoppingFrames;
};
}  // namespace Input
}  // namespace VideoStitch
