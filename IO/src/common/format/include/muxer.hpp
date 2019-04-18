// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "util.hpp"
#include "iopacket.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#define LIBAV_WRITER_DEFAULT_CONTAINER "mp4"

struct AVCodec;
struct AVCodecParameters;
struct AVFormatContext;
struct AVPacket;
struct AVStream;
struct AVDictionary;

namespace VideoStitch {
/**
 * Packets queue
 */
struct PacketQueue {
  PacketQueue(size_t streamSize) : streams(streamSize) {}

  void pushPacket(std::shared_ptr<struct AVPacket> pkt, const int streamId);
  void rescaleTimestamp(AVCodecContext* codecCtx, AVStream*, AVPacket*);

  std::shared_ptr<struct AVPacket> popPacket();
  bool isEmpty() const;

  size_t size() { return packets.size(); }

  std::vector<AVStream*> streams;

  std::queue<std::shared_ptr<struct AVPacket> > packets;

  std::mutex mutex;
  std::condition_variable cond;
  bool shutDown = false;
};

struct AVEncoder {
  int id;
  AVCodec* codec;
  AVCodecContext* codecContext;

  AVEncoder() : id(-1), codec(nullptr), codecContext(nullptr) {}

  AVEncoder(int id) : id(id), codec(nullptr), codecContext(nullptr) {}

  AVEncoder(int id, AVCodec* cdc, AVCodecContext* ctx) : id(id), codec(cdc), codecContext(ctx) {}
};

namespace Output {

/**
 * A wrapper around libavformat's muxers that lives in its own thread.
 */
class Muxer {
 public:
  explicit Muxer(size_t index, const std::string& format, std::vector<AVEncoder>& encoders, const AVDictionary* config);
  virtual ~Muxer();

  virtual void writeTrailer();

  int64_t getMuxedSize() const;

  void start();
  void join();

  void setThreadStatus(const MuxerThreadStatus status) { m_threadStatus = status; }
  MuxerThreadStatus getThreadStatus(void) const { return m_threadStatus; }

  AVFormatContext* formatCtx;
  AVDictionary* m_config;

  PacketQueue packets;

 protected:
  void writerGlobalHeaders();

 private:
  void writeHeader();
  void writeFrame(AVPacket* const);
  void run();

  // m_threadStatus is atomic because it can be set/get from different threads
  std::atomic<MuxerThreadStatus> m_threadStatus;
  std::thread thread;

  const size_t index;

  std::unordered_map<int, AVCodecContext*> encoderContexts;
};

}  // namespace Output
}  // namespace VideoStitch
