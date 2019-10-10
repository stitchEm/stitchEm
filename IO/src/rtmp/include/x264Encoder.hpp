// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoEncoder.hpp"
#include "amfIncludes.hpp"

extern "C" {
#if defined(_WIN32)
#include "x264.h"
#else
#include <inttypes.h>
#include <x264.h>
#endif
}

//#include "sps_decode.h"

/*
The different types of Network Abstraction Layer Units for reference:

0      Unspecified                                                    non - VCL
1      Coded slice of a non - IDR picture                             VCL
2      Coded slice data partition A                                   VCL
3      Coded slice data partition B                                   VCL
4      Coded slice data partition C                                   VCL
5      Coded slice of an IDR picture                                  VCL
6      Supplemental enhancement information(SEI)                      non - VCL
7      Sequence parameter set                                         non - VCL
8      Picture parameter set                                          non - VCL
9      Access unit delimiter                                          non - VCL
10     End of sequence                                                non - VCL
11     End of stream                                                  non - VCL
12     Filler data                                                    non - VCL
13     Sequence parameter set extension                               non - VCL
14     Prefix NAL unit                                                non - VCL
15     Subset sequence parameter set                                  non - VCL
16     Depth parameter set                                            non - VCL
17..18 Reserved                                                       non - VCL
19     Coded slice of an auxiliary coded picture without partitioning non - VCL
20     Coded slice extension                                          non - VCL
21     Coded slice extension for depth view components                non - VCL
22..23 Reserved                                                       non - VCL
24..31 Unspecified                                                    non - VCL
*/

namespace VideoStitch {
namespace Output {

class X264Encoder : public VideoEncoder {
 public:
  X264Encoder(FrameRate fps, int width, int height, const std::string& preset, const std::string& tune,
              const std::string& profile, const std::string& level, const std::string& bitrate_mode,
              int quality_balance, const VideoStitch::IO::ColorDescription& colorDesc, int maxBitrate, int bufferSize,
              bool cbr_padding, int gop, int b_frames);

  ~X264Encoder();

  static void x264_log(void*, int i_level, const char* psz, va_list args);
  static int getLevelFromString(const std::string& levelStr);

  static Potential<VideoEncoder> createX264Encoder(const Ptv::Value& config, int width, int height,
                                                   FrameRate framerate);

  bool encode(const Frame& videoFrame, std::vector<VideoStitch::IO::DataPacket>& packets);

  char* metadata(char* enc, char* pend);

  /**
   * Supplemental Enhancement Information unit
   */
  void getSEI(VideoStitch::IO::DataPacket& packet) { packet = sei; }

  int getBitRate() const;

  bool dynamicBitrateSupported() const { return (paramData.i_nal_hrd != X264_NAL_HRD_CBR); }

  bool setBitRate(uint32_t maxBitrate, uint32_t bufferSize);

 private:
  static const AVal av_videocodecid;
  static const AVal av_videodatarate;
  static const AVal av_framerate;

  void setBitRateParams(uint32_t maxBitrate, uint32_t bufferSize);

  x264_param_t paramData;
  x264_t* x264;

  x264_picture_t picOut;

  bool useCBR;
  bool doRequestKeyframe = true;

  unsigned int width, height;
  FrameRate framerate;

  std::string curPreset, curTune, curProfile;

  x264_picture_t pic;
};

}  // namespace Output
}  // namespace VideoStitch
