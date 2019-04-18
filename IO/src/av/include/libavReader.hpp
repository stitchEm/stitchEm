// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"
#include "libvideostitch/inputFactory.hpp"

#include <deque>
#include <chrono>

extern "C" {
#include <libavutil/pixfmt.h>
#ifdef SUP_QUICKSYNC
#include <libavcodec/qsv.h>
#endif
}
#undef PixelFormat

struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVFrame;
struct AVPacket;

class CHWDevice;
class D3DFrameAllocator;

static const int INVALID_STREAM_ID(-1);

namespace VideoStitch {

namespace Util {
class TimeoutHandler;
enum AvErrorCode : short;
}  // namespace Util

namespace Input {

#ifdef QUICKSYNC
class QSVContext {
 public:
  bool initMFX();
  static int getQSVBuffer(AVCodecContext* avctx, AVFrame* frame, int flags);
  static void freeQSVBuffer(void* opaque, uint8_t* data);

  mfxSession session;
  CHWDevice* hwdev;
  D3DFrameAllocator* allocator;

  mfxMemId* surface_ids;
  int* surface_used;
  int nb_surfaces;

  mfxFrameInfo frame_info;
};
#endif

/**
 * A deleter that does nothing
 */
template <class T>
struct NoopDeleter {
  NoopDeleter() {}
  NoopDeleter(const NoopDeleter<T>& /*other*/) {}
  void operator()(T*) const {}
};

/**
 * libav image reader.
 */
class LibavReader : public VideoReader, public AudioReader {
 public:
  // TODOLATERSTATUS replace by Input::ReadStatus
  enum class LibavReadStatus { Ok, EndOfPackets, Error };

  static ProbeResult probe(const std::string& fileNameTemplate);

  // ~ is protected, can't use Potential's DefaultDeleter
  typedef Potential<LibavReader, NoopDeleter<LibavReader>> PotentialLibavReader;

  static PotentialLibavReader create(const std::string& fileNameTemplate,
                                     VideoStitch::Plugin::VSReaderPlugin::Config runtime);

  virtual ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) override;
  virtual Status seekFrame(frameid_t) override;
  Status seekFrame(mtime_t) override;
  virtual size_t available() override;
  bool eos() override;

 protected:
  LibavReader(const std::string& displayName, const int64_t width, const int64_t height, const frameid_t firstFrame,
              const AVPixelFormat fmt, AddressSpace addrSpace, struct AVFormatContext* formatCtx,
#ifdef QUICKSYNC
              class QSVContext* qsvCtx,
#endif
              struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
              struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* video, struct AVFrame* audio,
              Util::TimeoutHandler* interruptCallback, const int videoIdx, const int audioIdx,
              const Audio::ChannelLayout layout, const Audio::SamplingRate samplingRate,
              const Audio::SamplingDepth samplingDepth);
  ~LibavReader();

  LibavReadStatus readPacket(AVPacket* pkt);

  static void findAvStreams(struct AVFormatContext* formatCtx, int& videoIdx, int& audioIdx);
  static enum AVPixelFormat selectFormat(struct AVCodecContext*, const enum AVPixelFormat*);

  void decodeVideoPacket(bool* got_picture, AVPacket* pkt, unsigned char* frame, bool flush = false);
  void flushVideoDecoder(bool* got_picture, unsigned char* frame);
  void decodeAudioPacket(AVPacket* pkt, bool flush = false);

  struct AVFormatContext* formatCtx;
#ifdef QUICKSYNC
  QSVContext* qsvCtx;
#endif
  struct AVCodecContext* videoDecoderCtx;
  struct AVCodecContext* audioDecoderCtx;

  const struct AVCodec* videoCodec;
  const struct AVCodec* audioCodec;

  struct AVFrame* videoFrame;
  struct AVFrame* audioFrame;

  Util::TimeoutHandler* interruptCallback;
  const int videoIdx;
  const int audioIdx;

  // time code of the last decoded video frame,
  // expressed in time_base units (eg. 1/90000 second),
  // from the start of the container (see start_time semantics)
  int64_t currentVideoPts;

  // time code of the first video frame, in container clock
  // in libvideostitch loadFrame date, this is 0
  int64_t firstVideoFramePts;

  std::vector<std::deque<uint8_t>> audioBuffer;
  size_t nbSamplesInAudioBuffer;
  mtime_t videoTimeStamp;
  mtime_t audioTimeStamp;

  bool expectingIncreasingVideoPts;

 private:
  static Util::AvErrorCode avDecodePacket(AVCodecContext* s, AVPacket* pkt, AVFrame* frame, bool* got_frame,
                                          bool flush = false);
  static int getBuffer(AVCodecContext* s, AVFrame* pic);
  static void releaseBuffer(AVCodecContext* /*s*/, AVFrame* pic);
};
}  // namespace Input
}  // namespace VideoStitch
