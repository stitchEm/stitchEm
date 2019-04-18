// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoEncoder.hpp"
#include "amfIncludes.hpp"
#include "NvVideoEncoder.h"

#include <cuda_runtime.h>

#include <queue>

#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024

#if defined __linux__
#define stricmp strcasecmp
#elif defined(_WIN32) || defined(_WIN64)
// The POSIX name for stricmp is deprecated. Instead, use the ISO C++ conformant name: _stricmp.
#define stricmp _stricmp
#endif

namespace VideoStitch {
namespace Output {

typedef struct _EncodeConfig {
  int width;
  int height;
  FrameRate fps;
  int bitrate;
  int vbvSize;
  enum v4l2_mpeg_video_bitrate_mode rcMode;
  int profile;
  FILE* fOutput;
  int encoder_pixfmt;
  int gopLength;
  int numB;
  int preset;

  std::string encoderProfile;
  std::string encoderLevel;

} EncodeConfig;

/**
 * The NvV4l2Encoder is based on V4L2, and supported on
 * NVIDIA tegra cards.
 */

class NvV4l2Encoder : public VideoEncoder {
 public:
  NvV4l2Encoder(EncodeConfig&);
  ~NvV4l2Encoder();

  static Potential<VideoEncoder> createNvV4l2Encoder(const Ptv::Value& config, int width, int height,
                                                     FrameRate framerate);
  static void supportedEncoders(std::vector<std::string>& codecs);

  bool encode(const Frame&, std::vector<VideoStitch::IO::DataPacket>&);
  bool ProcessOutput(std::vector<VideoStitch::IO::DataPacket>& packets);
  void getHeaders(VideoStitch::IO::DataPacket& packet);

  char* metadata(char* enc, char* pend);
  int getBitRate() const { return ctx.bitrate; }
  bool dynamicBitrateSupported() const { return false; }
  bool setBitRate(uint32_t /*maxBitrate*/, uint32_t /*bufferSize*/) { return false; }

 private:
  Status Initialize();
  int getProfile(const char*, int);
  enum v4l2_mpeg_video_h264_level getLevel(const char*, int);
  static bool encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer,
                                                void* arg);

  int setPreset(uint32_t preset);

  static const AVal av_videocodecid;
  static const AVal av_videodatarate;
  static const AVal av_framerate;

  EncodeConfig ctx;

  int index;
  NvVideoEncoder* enc;

  std::queue<mtime_t> timestamps;
  int first_dts;
  cudaStream_t stream;

  std::deque<std::pair<struct v4l2_buffer, NvBuffer*>>* queue;
  std::mutex queue_lock;
};

}  // namespace Output
}  // namespace VideoStitch
