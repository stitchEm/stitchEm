// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/profile.hpp"
#include "muxer.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdint.h>
#include <cstdio>

#ifndef _MSC_VER
#include <sys/time.h>
#endif

struct AVCodecContext;
struct AVFrame;

namespace VideoStitch {
namespace Util {
enum AvErrorCode : short;
}

namespace Output {

/**
 * @brief Additional indirection onto the implementation.
 * @note  Allows reseting the writer implementation while keeping the same LibavWriter object
 */
class AvMuxer_pimpl;

class LibavWriter : public VideoWriter, public AudioWriter {
 public:
  static Output* create(const Ptv::Value& config, const std::string& name, const char* baseName, unsigned width,
                        unsigned height, FrameRate framerate, const Audio::SamplingRate samplingRate,
                        const Audio::SamplingDepth samplingDepth, const Audio::ChannelLayout channleLayout);

  ~LibavWriter();

  void pushVideo(const Frame& videoFrame);
  void pushAudio(Audio::Samples& audioSamples);

 private:
  LibavWriter(const Ptv::Value& config, const std::string& name, const VideoStitch::PixelFormat fmt, AddressSpace type,
              unsigned width, unsigned height, FrameRate framerate, const Audio::SamplingRate samplingRate,
              const Audio::SamplingDepth samplingDepth, const Audio::ChannelLayout channleLayout);

  Util::AvErrorCode encodeVideoFrame(AVFrame* frame, int64_t frameOffset);
  Util::AvErrorCode encodeAudioFrame(AVFrame* frame);
  MuxerThreadStatus flushVideo();
  MuxerThreadStatus flushAudio();
  MuxerThreadStatus close();

  bool needsRespawn(std::shared_ptr<AvMuxer_pimpl>&, mtime_t);
  bool implReady(std::shared_ptr<AvMuxer_pimpl>&, AVCodecContext*, mtime_t);
  bool hasAudio() const { return audioCodecContext != nullptr; }

  bool createVideoCodec(AddressSpace type, unsigned width, unsigned height, FrameRate framerate);
  bool createAudioCodec();
  bool resetCodec(AVCodecContext*, MuxerThreadStatus& status);
  Ptv::Value* m_config;

  std::deque<AVFrame*> videoFrames;
  AVDictionary* codecConfig;
  AVCodecContext* videoCodecContext;
  mtime_t firstVideoPTS;

  AVFrame* audioFrame;
  Audio::Samples audioBuffer;
  uint8_t* audioData[MAX_AUDIO_CHANNELS];  // intermediate buffer
  const uint8_t* avSamples;                // buffer used by libav
  Audio::SamplingFormat m_sampleFormat;
  std::vector<int64_t> m_channelMap;
  std::size_t m_audioFrameSizeInBytes;
  AVCodecContext* audioCodecContext;

  int m_currentImplNumber;
  std::shared_ptr<AvMuxer_pimpl> m_pimplVideo;
  std::shared_ptr<AvMuxer_pimpl> m_pimplAudio;
  std::mutex pimplMu;
};
}  // namespace Output
}  // namespace VideoStitch
