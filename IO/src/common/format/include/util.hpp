// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/input.hpp"

extern "C" {
/* TODO : to be removed when apple use ffmpeg3.2 */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#endif
#include <libavutil/samplefmt.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

struct AVCodecContext;
struct AVDictionary;
struct AVFormatContext;

namespace VideoStitch {
namespace Util {

enum AvErrorCode : short { Ok, TryAgain, EndOfFile, InvalidRequest, GenericError };

/**
 * Handles libav stuffs common to reader and writer.
 */
class Libav {
 public:
  /**
   * Checks that libav is initialized. If not, performs the
   * initialization.
   *
   * Note: this holds a lock. So their must be no other Lock at the
   * same time.
   */
  static void checkInitialization();

  /**
   * An RAII lock.
   *
   * libav is not thread-safe. That means that when its non-reentrant
   * functions of are used, you need to lock to avoid concurrent
   * calls.
   */
  class Lock {
   public:
    /**
     * Holds a lock.
     *
     * The following functions of libav are not reentrant:
     *   avcodec_open()
     *   avcodec_close()
     *   avformat_find_stream_info()
     */
    Lock();

    /**
     * Unlocks.
     */
    ~Lock();
  };
};

class TimeoutHandler {
 public:
  using duration = std::chrono::milliseconds;

#if defined(_MSC_VER) && _MSC_VER <= 1800
  using clock = std::chrono::system_clock;
#else
  using clock = std::chrono::steady_clock;
#endif

  using time_point = std::chrono::time_point<clock, duration>;

  explicit TimeoutHandler(std::chrono::milliseconds timeout) : timeout(computeTimeout(timeout)) {}

  void reset(std::chrono::milliseconds t) { timeout = computeTimeout(t); }

  static int checkInterrupt(void *t) { return t && static_cast<const TimeoutHandler *>(t)->isTimeout(); }

 private:
  static time_point computeTimeout(std::chrono::milliseconds timeout) {
    return std::chrono::time_point_cast<duration>(clock::now()) + timeout;
  }

  static void logTimeout();

  bool isTimeout() const {
#ifndef __ANDROID__
    bool timedOut = clock::now() > timeout.load();
#else
    bool timedOut = clock::now() > timeout;
#endif
    if (timedOut) {
      logTimeout();
    }
    return timedOut;
  }

#ifndef __ANDROID__
  std::atomic<time_point> timeout;
#else
  time_point timeout;
#endif
};

void build_dict(AVDictionary **dict, const char *options, const char *type);

void split(const char *str, char delim, std::vector<std::string> *res);

const std::string errorString(const int errorCode);

bool isStream(const std::string &filename);

AVSampleFormat libavSampleFormat(const Audio::SamplingDepth depth);

VideoStitch::Audio::SamplingDepth sampleFormat(const AVSampleFormat depth);

VideoStitch::Audio::ChannelLayout channelLayout(const uint64_t layout);

VideoStitch::Audio::ChannelMap getChannelMap(int avChannelIndex, VideoStitch::Audio::ChannelLayout layout);

AvErrorCode getAvErrorCode(const int errorCode);

const std::string errorStringFromAvErrorCode(const AvErrorCode errorCode);
}  // namespace Util
}  // namespace VideoStitch
