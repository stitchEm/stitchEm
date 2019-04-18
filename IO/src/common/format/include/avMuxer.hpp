// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/ptv.hpp"
#include "libvideostitch/frame.hpp"

#include "iopacket.hpp"

#include <atomic>
#include <memory>
#include <vector>

#ifndef _MSC_VER
#include <sys/time.h>
#endif

#define LIBAV_WRITER_DEFAULT_FRAMERATE 25.0
#define LIBAV_WRITER_LIVE_RTMP "rtmp"
#define LIBAV_WRITER_DEFAULT_CODEC "h264"
#define LIBAV_WRITER_DEFAULT_BITRATE 15000000   // Bps (15mbps)
#define LIBAV_WRITER_MAX_MP4_BITRATE 110000000  // Bps (110mbps)
#define LIBAV_WRITER_MIN_MP4_BITRATE 300000     // Bps (300kps)
#define LIBAV_WRITER_DEFAULT_BITRATE_MODE "VBR"
#define LIBAV_WRITER_DEFAULT_GOP_SIZE ((int)(1.0 * LIBAV_WRITER_DEFAULT_FRAMERATE))
#define LIBAV_WRITER_MAX_GOP_SIZE ((int)(10.0 * LIBAV_WRITER_DEFAULT_FRAMERATE))
#define LIBAV_WRITER_DEFAULT_B_FRAMES 2
#define LIBAV_WRITER_MAX_B_FRAMES 5
#define LIBAV_WRITER_DEFAULT_NUM_PASS 1
#define LIBAV_WRITER_MIN_QSCALE 1
#define LIBAV_WRITER_MAX_QSCALE 31
#define LIBAV_DEFAULT_AUDIO_BITRATE 128  // Bits (128kbps)
#define LIBAV_BUFFER_ALIGNMENT 1         // No align

struct AVCodec;
struct AVCodecContext;
struct AVEncoder;

namespace VideoStitch {
namespace Output {

class Muxer;

std::shared_ptr<struct AVPacket> newPacket();

/**
 * @brief Additional indirection onto the implementation.
 * @note  Allows reseting the writer implementation while keeping the same LibavWriter object
 */
class AvMuxer_pimpl {
 public:
  /* create from AVCodecContext */
  static AvMuxer_pimpl* create(const Ptv::Value& config, AVCodecContext* videoCodecCtx, AVCodecContext* audioCodecCtx,
                               FrameRate framerate, int currentImplNumber);

  /* create without AVCodecContext */
  static AvMuxer_pimpl* create(const Ptv::Value& config, std::vector<AVEncoder>& codecs, unsigned width,
                               unsigned height, FrameRate framerate, mtime_t firstPTS, int currentImplNumber);

  AvMuxer_pimpl(const int64_t maxMuxedSize, const int maxFrameId, mtime_t firstPTS, const std::vector<Muxer*>& muxers);

  ~AvMuxer_pimpl();

  void pushVideoPacket(const std::shared_ptr<struct AVPacket> pkt);
  void pushAudioPacket(const std::shared_ptr<struct AVPacket> pkt);
  void pushMetadataPacket(const std::shared_ptr<struct AVPacket> pkt);
  void pushVideoPacket(const VideoStitch::IO::Packet& pkt);
  void pushAudioPacket(const VideoStitch::IO::Packet& pkt);
  void pushMetadataPacket(const VideoStitch::IO::Packet& pkt);

  MuxerThreadStatus getStatus();

  bool needsRespawn() const;

  bool firstFrame() const { return m_firstPTS == -1; }

  /**
   * @brief Flush frames, and close encoders
   */
  MuxerThreadStatus close();

  friend class LibavWriter;
  friend class AvMuxer;

 private:
  static AvMuxer_pimpl* create_priv(const Ptv::Value& config, std::vector<AVEncoder>& codecs, unsigned width,
                                    unsigned height, FrameRate framerate, mtime_t firstPTS, int currentImplNumber);
  void start();

  /**
   * @brief update m_needsRespawn field
   */
  void updateRespawnStatus(const bool isvideo);

  std::vector<Muxer*> muxers;

  const int64_t m_maxMuxedSize;
  const int m_maxFrameId;
  int m_curFrameId;
  std::atomic<MuxerThreadStatus> m_Status;
  std::atomic<bool> m_needsRespawn;
  mtime_t m_firstPTS;
  mtime_t m_lastPTS;
};

class AvMuxer {
 public:
  AvMuxer();

  ~AvMuxer();

  VideoStitch::Status create(const Ptv::Value& config, unsigned width, unsigned height, FrameRate framerate,
                             Span<unsigned char>& header, mtime_t videotimestamp, mtime_t audiotimestamp);

  void destroy();

  void pushVideoPacket(const VideoStitch::IO::Packet& pkt);
  void pushAudioPacket(const VideoStitch::IO::Packet& pkt);
  void pushMetadataPacket(const VideoStitch::IO::Packet& pkt);

 private:
  bool implReady();

  std::vector<AVEncoder> m_codecs;

  Ptv::Value* m_config;
  unsigned m_width;
  unsigned m_height;
  FrameRate m_framerate;
  std::vector<unsigned char> m_header;
  mtime_t m_firstPTS;
  mtime_t m_audioOffset;

  int m_sampleRate;
  Audio::SamplingDepth m_sampleDepth;
  uint64_t m_channelLayout;

  int m_currentImplNumber;
  std::unique_ptr<AvMuxer_pimpl> m_pimpl;
  std::mutex pimplMu;
};

}  // namespace Output
}  // namespace VideoStitch
