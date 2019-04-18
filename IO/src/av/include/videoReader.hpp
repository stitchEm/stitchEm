// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libavReader.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

#include <mutex>

struct AVFormatContext;
struct AVCodec;
struct AVFrame;

namespace VideoStitch {
namespace Input {
/**
 * libav image reader.
 */
class FFmpegReader : public LibavReader {
 public:
  enum FFmpegReadStatus { Ok, Error, Continue, EOS };

  static bool handles(const std::string& filename);

  Status seekFrame(frameid_t) override;
  ReadStatus readFrame(mtime_t& date, unsigned char* videoFrame) override;
  ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) override;
  size_t available() override;

  virtual ~FFmpegReader();

  FFmpegReader(readerid_t id, const std::string& displayName, const int64_t width, const int64_t height,
               const int firstFrame, const AVPixelFormat fmt, AddressSpace addrSpace, struct AVFormatContext* formatCtx,
               struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
               struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* videoFrame,
               struct AVFrame* audioFrame, Util::TimeoutHandler* interruptCallback, const int videoIdx,
               const int audioIdx, const Audio::ChannelLayout layout, const Audio::SamplingRate samplingRate,
               const Audio::SamplingDepth samplingDepth);

 private:
  bool ensureAudio(size_t nbSamples);

  std::vector<unsigned char> frame;
  std::recursive_mutex monitor;
  std::deque<AVPacket> videoQueue, audioQueue;  // for audio preroll
};
}  // namespace Input
}  // namespace VideoStitch
