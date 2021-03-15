// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "util.hpp"

#include "libvideostitch/logging.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <assert.h>
#include <algorithm>
#include <functional>
#include <mutex>
#include <iostream>
#include <vector>
#include <regex>

namespace VideoStitch {
namespace Util {
using namespace Audio;

static const size_t ERROR_BUFFER_SIZE(64);

namespace {
bool isLibavInit = false;  // protected by Util::GlobalLock lock;
std::mutex libAvMutex;

ThreadSafeOstream& av2vs_log(int level) {
  switch (level) {
    case AV_LOG_QUIET:
      return VideoStitch::Logger::get(VideoStitch::Logger::Quiet);
    case AV_LOG_PANIC:
    case AV_LOG_FATAL:
    case AV_LOG_ERROR:
      return VideoStitch::Logger::get(VideoStitch::Logger::Warning);
    case AV_LOG_WARNING:
      return VideoStitch::Logger::get(VideoStitch::Logger::Info);
    case AV_LOG_INFO:
      return VideoStitch::Logger::get(VideoStitch::Logger::Verbose);
    case AV_LOG_VERBOSE:
    case AV_LOG_DEBUG:
      return VideoStitch::Logger::get(VideoStitch::Logger::Debug);
    default:
      return VideoStitch::Logger::get(VideoStitch::Logger::Debug);
  }
}

const char* avlog_name(int level) {
  switch (level) {
    case AV_LOG_QUIET:
      return "quiet";
    case AV_LOG_PANIC:
      return "panic";
    case AV_LOG_FATAL:
      return "fatal";
    case AV_LOG_ERROR:
      return "error";
    case AV_LOG_WARNING:
      return "warning";
    case AV_LOG_INFO:
      return "info";
    case AV_LOG_VERBOSE:
      return "verbose";
    case AV_LOG_DEBUG:
      return "debug";
    default:
      return "unknown";
  }
}

void vs_avlog(void* /*avcl*/, int level, const char* fmt, va_list vl) {
  char buffer[1024];
  if (level > av_log_get_level()) {
    return;  //  drop the message
  }
  auto& log = av2vs_log(level);
  vsnprintf(buffer, sizeof(buffer) - 1, fmt, vl);
  log << "[libav-" << avlog_name(level) << "] " << buffer << std::flush;
}
}  // namespace

Libav::Lock::Lock() { libAvMutex.lock(); }

Libav::Lock::~Lock() { libAvMutex.unlock(); }

void Libav::checkInitialization() {
  Lock lock;
  if (!isLibavInit) {
    av_register_all();
    avformat_network_init();
    isLibavInit = true;
    /* custom log */
    av_log_set_level(AV_LOG_WARNING);
    av_log_set_callback(vs_avlog);
  }
}

void split(const char* str, char delim, std::vector<std::string>* res) {
  const char* lastP = str;
  for (const char* p = str; *p != '\0'; ++p) {
    if (*p == delim) {
      res->push_back(std::string(lastP, p - lastP));
      lastP = p + 1;
    }
  }
  res->push_back(std::string(lastP));
}

void build_dict(AVDictionary** dict, const char* options, const char* type) {
  char* opt = strdup(options);
  char* tok = strtok(opt, "- ");
  char* tokval = nullptr;
  while (tok && (tokval = strtok(nullptr, "- "))) {
    if (av_dict_set(dict, tok, tokval, 0) < 0)
      VideoStitch::Logger::get(VideoStitch::Logger::Error)
          << "[libavoutput] unknown" << type << "option \"" << tok << "\" with value \"" << tokval << "\"" << std::endl;
    tok = strtok(nullptr, "- ");
  }
  free(opt);
}

const std::string errorString(const int errorCode) {
  char err[ERROR_BUFFER_SIZE];
  av_strerror(errorCode, err, ERROR_BUFFER_SIZE);
  return err;
}

#ifndef __clang_analyzer__
bool isStream(const std::string& filename) {
  std::regex rtsp("RTSP://", std::regex_constants::ECMAScript | std::regex_constants::icase);
   std::regex rtmp("RTMP://", std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(filename, rtsp)) {
    return true;
  } else if (std::regex_search(filename, rtmp)) {
    return true;
  }
  return false;
}
#endif  // __clang_analyzer__

AVSampleFormat libavSampleFormat(const Audio::SamplingDepth depth) {
  switch (depth) {
    case Audio::SamplingDepth::UINT8:
      return AV_SAMPLE_FMT_U8;
    case Audio::SamplingDepth::INT16:
      return AV_SAMPLE_FMT_S16;
    case Audio::SamplingDepth::INT32:
      return AV_SAMPLE_FMT_S32;
    case Audio::SamplingDepth::FLT:
      return AV_SAMPLE_FMT_FLT;
    case Audio::SamplingDepth::DBL:
      return AV_SAMPLE_FMT_DBL;
    case Audio::SamplingDepth::UINT8_P:
      return AV_SAMPLE_FMT_U8P;
    case Audio::SamplingDepth::INT16_P:
      return AV_SAMPLE_FMT_S16P;
    case Audio::SamplingDepth::INT32_P:
      return AV_SAMPLE_FMT_S32P;
    case Audio::SamplingDepth::FLT_P:
      return AV_SAMPLE_FMT_FLTP;
    case Audio::SamplingDepth::DBL_P:
      return AV_SAMPLE_FMT_DBLP;
    default:
      return AV_SAMPLE_FMT_NONE;
  }
}

Audio::SamplingDepth sampleFormat(const AVSampleFormat depth) {
  switch (depth) {
    case AV_SAMPLE_FMT_U8:
      return Audio::SamplingDepth::UINT8;
    case AV_SAMPLE_FMT_S16:
      return Audio::SamplingDepth::INT16;
    case AV_SAMPLE_FMT_S32:
      return Audio::SamplingDepth::INT32;
    case AV_SAMPLE_FMT_FLT:
      return Audio::SamplingDepth::FLT;
    case AV_SAMPLE_FMT_DBL:
      return Audio::SamplingDepth::DBL;
    case AV_SAMPLE_FMT_U8P:
      return Audio::SamplingDepth::UINT8_P;
    case AV_SAMPLE_FMT_S16P:
      return Audio::SamplingDepth::INT16_P;
    case AV_SAMPLE_FMT_S32P:
      return Audio::SamplingDepth::INT32_P;
    case AV_SAMPLE_FMT_FLTP:
      return Audio::SamplingDepth::FLT_P;
    case AV_SAMPLE_FMT_DBLP:
      return Audio::SamplingDepth::DBL_P;
    case AV_SAMPLE_FMT_NB:
    case AV_SAMPLE_FMT_NONE:
      return Audio::SamplingDepth::SD_NONE;
    default:
      return Audio::SamplingDepth::SD_NONE;
  }
}

ChannelLayout channelLayout(const uint64_t layout) {
  switch (layout) {
    case AV_CH_LAYOUT_MONO:
      return MONO;
    case AV_CH_LAYOUT_STEREO:
      return STEREO;
    case AV_CH_LAYOUT_2POINT1:
      return _2POINT1;
    case AV_CH_LAYOUT_2_1:
      return _2_1;
    case AV_CH_LAYOUT_SURROUND:
      return SURROUND;
    case AV_CH_LAYOUT_3POINT1:
      return _3POINT1;
    case AV_CH_LAYOUT_4POINT0:
      return _4POINT0;
    case AV_CH_LAYOUT_4POINT1:
      return _4POINT1;
    case AV_CH_LAYOUT_2_2:
      return _2_2;
    case AV_CH_LAYOUT_QUAD:
      return QUAD;
    case AV_CH_LAYOUT_5POINT0:
      return _5POINT0;
    case AV_CH_LAYOUT_5POINT1:
      return _5POINT1;
    case AV_CH_LAYOUT_5POINT0_BACK:
      return _5POINT0_BACK;
    case AV_CH_LAYOUT_5POINT1_BACK:
      return _5POINT1_BACK;
    case AV_CH_LAYOUT_6POINT0:
      return _6POINT0;
    case AV_CH_LAYOUT_6POINT0_FRONT:
      return _6POINT0_FRONT;
    case AV_CH_LAYOUT_HEXAGONAL:
      return HEXAGONAL;
    case AV_CH_LAYOUT_6POINT1:
      return _6POINT1;
    case AV_CH_LAYOUT_6POINT1_BACK:
      return _6POINT1_BACK;
    case AV_CH_LAYOUT_6POINT1_FRONT:
      return _6POINT1_FRONT;
    case AV_CH_LAYOUT_7POINT0:
      return _7POINT0;
    case AV_CH_LAYOUT_7POINT0_FRONT:
      return _7POINT0_FRONT;
    case AV_CH_LAYOUT_7POINT1:
      return _7POINT1;
    case AV_CH_LAYOUT_7POINT1_WIDE:
      return _7POINT1_WIDE;
    case AV_CH_LAYOUT_7POINT1_WIDE_BACK:
      return _7POINT1_WIDE_BACK;
    case AV_CH_LAYOUT_OCTAGONAL:
      return OCTAGONAL;
    case AV_CH_LAYOUT_STEREO_DOWNMIX:
      return STEREO;
    default:
      return UNKNOWN;
  }
}

// Returns the Channel Map corrsponding of the i-th channel in the layout
// For example, for the layout _5POINT1 :
// getChannelMap(0, _5POINT1) returns SPEAKER_FRONT_LEFT
// getChannelMap(1, _5POINT1) returns SPEAKER_FRONT_RIGHT
// getChannelMap(2, _5POINT1) returns SPEAKER_SIDE_LEFT
// getChannelMap(3, _5POINT1) returns SPEAKER_SIDE_RIGHT
// getChannelMap(4, _5POINT1) returns SPEAKER_FRONT_CENTER
// getChannelMap(5, _5POINT1) returns SPEAKER_LOW_FREQUENCY
ChannelMap getChannelMap(int avChannelIndex, ChannelLayout layout) {
  int channelIndex = 0;
  for (int c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
    if (getChannelMapFromChannelIndex(c) & layout) {
      if (channelIndex == avChannelIndex) {
        return getChannelMapFromChannelIndex(c);
      }
      channelIndex++;
    }
  }
  return NO_SPEAKER;
}

AvErrorCode getAvErrorCode(const int errorCode) {
  switch (errorCode) {
    case 0:
      return AvErrorCode::Ok;
    case AVERROR(EAGAIN):
      return AvErrorCode::TryAgain;
    case AVERROR_EOF:
      return AvErrorCode::EndOfFile;
    case AVERROR(EINVAL):
      return AvErrorCode::InvalidRequest;
    case AVERROR(ENOMEM):
    default:
      return AvErrorCode::GenericError;
  }
}

const std::string errorStringFromAvErrorCode(const AvErrorCode errorCode) {
  switch (errorCode) {
    case AvErrorCode::Ok:
      return "Ok";
    case AvErrorCode::TryAgain:
      return "Resource temporarily unavailable";
    case AvErrorCode::EndOfFile:
      return "End of file. Encoder/decoder fully flushed";
    case AvErrorCode::InvalidRequest:
      return "Codec not opened, or invalid request";
    case AvErrorCode::GenericError:
    default:
      return "Internal AV error";
  }
}

}  // namespace Util
}  // namespace VideoStitch
