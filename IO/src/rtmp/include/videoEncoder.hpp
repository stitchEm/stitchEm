// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "rtmpStructures.hpp"

#include "libvideostitch/ptv.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/status.hpp"

extern "C" {
#if defined(_WIN32)
#include "x264/x264.h"
#else
#include <unistd.h>
#include <inttypes.h>
#include <x264.h>
#endif
}

#include <vector>

typedef enum {
  NAL_UNIT_CODED_SLICE_TRAIL_N = 0,
  NAL_UNIT_CODED_SLICE_TRAIL_R,
  NAL_UNIT_CODED_SLICE_TSA_N,
  NAL_UNIT_CODED_SLICE_TLA_R,
  NAL_UNIT_CODED_SLICE_STSA_N,
  NAL_UNIT_CODED_SLICE_STSA_R,
  NAL_UNIT_CODED_SLICE_RADL_N,
  NAL_UNIT_CODED_SLICE_RADL_R,
  NAL_UNIT_CODED_SLICE_RASL_N,
  NAL_UNIT_CODED_SLICE_RASL_R,
  NAL_UNIT_CODED_SLICE_BLA_W_LP = 16,
  NAL_UNIT_CODED_SLICE_BLA_W_RADL,
  NAL_UNIT_CODED_SLICE_BLA_N_LP,
  NAL_UNIT_CODED_SLICE_IDR_W_RADL,
  NAL_UNIT_CODED_SLICE_IDR_N_LP,
  NAL_UNIT_CODED_SLICE_CRA,
  NAL_UNIT_VPS = 32,
  NAL_UNIT_SPS,
  NAL_UNIT_PPS,
  NAL_UNIT_ACCESS_UNIT_DELIMITER,
  NAL_UNIT_EOS,
  NAL_UNIT_EOB,
  NAL_UNIT_FILLER_DATA,
  NAL_UNIT_PREFIX_SEI,
  NAL_UNIT_SUFFIX_SEI,
  NAL_UNIT_INVALID = 64,
} NalUnitType;

typedef struct x265_nal {
  uint32_t type;      /* NalUnitType */
  uint32_t sizeBytes; /* size in bytes */
  uint8_t* payload;
} x265_nal;

#include "rtmpStructures.hpp"

namespace VideoStitch {
namespace Output {

class VideoEncoder {
 public:
  virtual ~VideoEncoder() {}

  static Potential<VideoEncoder> createVideoEncoder(const Ptv::Value& config, int width, int height,
                                                    FrameRate framerate, const std::string& encoderType);
  static void supportedCodecs(std::vector<std::string>& codecs);

  virtual char* metadata(char* ptr, char* ptrEnd) = 0;

  virtual int getBitRate() const = 0;
  virtual bool dynamicBitrateSupported() const = 0;
  virtual bool setBitRate(uint32_t bitrate, uint32_t bufferSize) = 0;
  int32_t getMaxBitRate() const { return bitrateMax; }
  void setMaxBitRate(const int32_t bitrate) { bitrateMax = bitrate; }

  virtual void header(std::vector<VideoStitch::IO::DataPacket>& packets);

  virtual bool encode(const Frame&, std::vector<VideoStitch::IO::DataPacket>& packets) = 0;

  static void createDataPacket(std::vector<x264_nal_t>&, std::vector<VideoStitch::IO::DataPacket>&, mtime_t pts,
                               mtime_t dts);
  static void createHEVCPacket(std::vector<x265_nal>&, std::vector<VideoStitch::IO::DataPacket>&, mtime_t pts,
                               mtime_t dts);

 protected:
  std::vector<unsigned char> headPkt, sei;
  int32_t bitrateMax;
};

}  // namespace Output
}  // namespace VideoStitch
