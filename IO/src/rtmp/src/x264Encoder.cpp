// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cstdio>
#include <sstream>
#include <string>

#include "libvideostitch/logging.hpp"

#include "librtmpIncludes.hpp"
#include "x264Encoder.hpp"
#include "ptvMacro.hpp"

namespace VideoStitch {
namespace Output {

const AVal X264Encoder::av_videocodecid = mAVC("videocodecid");
const AVal X264Encoder::av_videodatarate = mAVC("videodatarate");
const AVal X264Encoder::av_framerate = mAVC("framerate");
const int baseCRF = 32;

X264Encoder::X264Encoder(FrameRate fps, int width, int height, const std::string& preset, const std::string& tune,
                         const std::string& profile, const std::string& level, const std::string& bitrate_mode,
                         int quality_balance, const VideoStitch::IO::ColorDescription& colorDesc, int maxBitrate,
                         int bufferSize, bool cbr_padding, int gop, int b_frames)
    : useCBR(bitrate_mode == "CBR"),
      width(width),
      height(height),
      framerate(fps),
      curPreset(preset),
      curTune(tune),
      curProfile(profile) {
  memset(&paramData, 0, sizeof(paramData));
  if (x264_param_default_preset(&paramData, curPreset.c_str(), curTune.c_str())) {
    Logger::get(Logger::Warning) << "RTMP : Failed to set x264 defaults: " << curPreset << "/" << curTune << std::endl;
  }

  paramData.b_deterministic = false;

  this->bitrateMax = maxBitrate;

  setBitRateParams(maxBitrate, bufferSize);

  if (useCBR) {
    if (cbr_padding) {
      paramData.rc.b_filler = 1;
    }
    paramData.rc.i_rc_method = X264_RC_ABR;
    paramData.rc.f_rf_constant = 0.0f;
  } else {
    paramData.rc.i_rc_method = X264_RC_CRF;
    paramData.rc.f_rf_constant = float(baseCRF - quality_balance);
    /* clip parameter so that the other parameters can be applied */
    if ((quality_balance >= baseCRF) && (x264_param_apply_profile(&paramData, curProfile.c_str()))) {
      quality_balance = baseCRF - 1;
      paramData.rc.f_rf_constant = float(baseCRF - quality_balance);
      Logger::warning("RTMP") << "Lossless coding not supported : reducing quality_balance to " << quality_balance
                              << std::endl;
    }
  }

  paramData.i_width = width;
  paramData.i_height = height;

  paramData.vui.b_fullrange = colorDesc.fullRange;
  paramData.vui.i_colorprim = colorDesc.primaries;
  paramData.vui.i_transfer = colorDesc.transfer;
  paramData.vui.i_colmatrix = colorDesc.matrix;

  // Group Of Pictures parameters
  if (gop > 0) {
    paramData.i_keyint_min = gop;
    paramData.i_keyint_max = gop;
  } else {
    Logger::info("RTMP") << "RTMP : default automatic GOP range from " << paramData.i_keyint_min << " to "
                         << paramData.i_keyint_max << std::endl;
  }
  paramData.i_bframe = b_frames;

  paramData.i_fps_num = fps.num;
  paramData.i_fps_den = fps.den;

  // timebase is set to the millisecond
  paramData.i_timebase_num = 1;
  paramData.i_timebase_den = 1000;

  paramData.i_level_idc = getLevelFromString(level);
  paramData.pf_log = x264_log;
  paramData.i_log_level = X264_LOG_DEBUG;
  paramData.i_csp = X264_CSP_I420;

  if (x264_param_apply_profile(&paramData, curProfile.c_str())) {
    Logger::get(Logger::Error) << "RTMP : Failed to set x264 profile: " << curProfile << std::endl;
  }

  x264 = x264_encoder_open(&paramData);
  if (!x264) {
    Logger::get(Logger::Error) << "RTMP : Could not initialize x264" << std::endl;
    assert(false);
  }

  x264_picture_alloc(&pic, X264_CSP_I420, paramData.i_width, paramData.i_height);
}

X264Encoder::~X264Encoder() {
  x264_picture_clean(&pic);
  x264_encoder_close(x264);
}

void X264Encoder::x264_log(void*, int i_level, const char* psz, va_list args) {
  ThreadSafeOstream* log;
  switch (i_level) {
    case X264_LOG_ERROR:
      log = &Logger::get(Logger::Error);
      break;
    case X264_LOG_WARNING:
      log = &Logger::get(Logger::Warning);
      break;
    case X264_LOG_INFO:
      log = &Logger::get(Logger::Info);
      break;
    case X264_LOG_DEBUG:
    default:
      log = &Logger::get(Logger::Debug);
      break;
  }
  *log << "RTMP : ";
  char logLine[256];
  vsnprintf(logLine, 256, psz, args);
  *log << logLine;
}

int X264Encoder::getLevelFromString(const std::string& levelStr) {
  // See the definition of x264_levels[] in set.c of lib x264
  std::string::size_type index;
  int firstNumber = std::stoi(levelStr, &index);
  int secondNumber = 0;
  if (index < levelStr.size()) {
    secondNumber = std::stoi(levelStr.substr(index + 1));
  }
  return firstNumber * 10 + secondNumber;
}

bool X264Encoder::encode(const Frame& videoFrame, std::vector<VideoStitch::IO::DataPacket>& packets) {
  packets.clear();

  memcpy(pic.img.plane[0], videoFrame.planes[0], width * height);
  memcpy(pic.img.plane[1], videoFrame.planes[1], (width * height) / 4);
  memcpy(pic.img.plane[2], videoFrame.planes[2], (width * height) / 4);

  // timebase was set to the millisecond
  pic.i_pts = int64_t(std::round(videoFrame.pts / 1000));

  x264_nal_t* nalOut;
  int nalNum;

  if (doRequestKeyframe) pic.i_type = X264_TYPE_IDR;
  if (x264_encoder_encode(x264, &nalOut, &nalNum, &pic, &picOut) < 0) {
    Logger::get(Logger::Error) << "RTMP : x264 encode failed" << std::endl;
    return false;
  }
  if (doRequestKeyframe) {
    pic.i_type = X264_TYPE_AUTO;
    doRequestKeyframe = false;
  }

  // NALU into FLV format packets
  // https://en.wikipedia.org/wiki/Network_Abstraction_Layer
  // http://download.macromedia.com/f4v/video_file_format_spec_v10_1.pdf
  for (int i = 0; i < nalNum; i++) {
    x264_nal_t& nal = nalOut[i];

    if (nal.i_type == NAL_SLICE_IDR || nal.i_type == NAL_SLICE || nal.i_type == NAL_SEI) {
      int j = 0;
      int start_code_offset = (nal.p_payload[2] == 0x01) ? 3 : 4;
      size_t size = nal.i_payload - start_code_offset;

      VideoStitch::IO::DataPacket pkt(size + 9);
      pkt.timestamp = uint64_t(picOut.i_dts);

      // VideoTagHeader
      pkt[j++] = (nal.i_type == NAL_SLICE_IDR) ? 0x17 : 0x27;  // keyframe vs. I/P/B slice
      pkt[j++] = 0x01;
      // Composition time : See ISO 14496-12, 8.15.3
      // offset between decoding time and presentation/composition time
      uint64_t composition_time_offset = uint64_t(picOut.i_pts - pkt.timestamp);
      pkt[j++] = composition_time_offset >> 16 & 0xff;
      pkt[j++] = composition_time_offset >> 8 & 0xff;
      pkt[j++] = composition_time_offset & 0xff;

      // Annex-B to MP4
      // NAL unit size
      pkt[j++] = size >> 24 & 0xff;
      pkt[j++] = size >> 16 & 0xff;
      pkt[j++] = size >> 8 & 0xff;
      pkt[j++] = size & 0xff;
      // NAL unit data
      memcpy(&pkt[j], nal.p_payload + start_code_offset, size);
      switch (nal.i_ref_idc) {
        case NAL_PRIORITY_DISPOSABLE:
          pkt.type = VideoStitch::IO::PacketType_VideoDisposable;
          break;
        case NAL_PRIORITY_LOW:
          pkt.type = VideoStitch::IO::PacketType_VideoLow;
          break;
        case NAL_PRIORITY_HIGH:
          pkt.type = VideoStitch::IO::PacketType_VideoHigh;
          break;
        case NAL_PRIORITY_HIGHEST:
        default:
          pkt.type = VideoStitch::IO::PacketType_VideoHighest;
          break;
      }

      packets.push_back(pkt);
    } else if (nal.i_type == NAL_SPS) {
      unsigned char* sps = nal.p_payload + 4;
      int spsLen = nal.i_payload - 4;
      /* DEBUG
      int width, height, fps;
      h264_decode_sps(sps, spsLen, width, height, fps);
      */
      int j = 0;

      VideoStitch::IO::DataPacket::Storage storage = VideoStitch::IO::DataPacket::make_storage(1024);
      unsigned char* pkt = storage.get();

      pkt[j++] = 0x17;  // H.264 K frame
      pkt[j++] = 0x00;  // codec config packet

      // start time on 24 bits
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;

      // ISO/IEC 14496-15 AVCDecoderConfigurationRecord
      // http://www.nhzjj.com/asp/admin/editor/newsfile/201011314552121.pdf
      pkt[j++] = 0x01;    // configurationVersion
      pkt[j++] = sps[1];  // AVCProfileIndication should be in (66, 77, 88, 100, 120, ..)
      pkt[j++] = sps[2];  // profile_compatibility (generally 0)
      pkt[j++] = sps[3];  // AVCLevelIndication, <= 51
      // bit(6) reserved = '111111'b;
      // unsigned int(2) lengthSizeMinusOne; <- our length is encoded on 4 bytes on the slices NAL units
      pkt[j++] = 0xff;
      // bit(3) reserved = '111'b;
      // unsigned int(5) numOfSequenceParameterSets;
      pkt[j++] = 0xe1;  // => only 1 SPS
      // SPS
      pkt[j++] = (spsLen >> 8) & 0xff;
      pkt[j++] = spsLen & 0xff;
      memcpy(&pkt[j], sps, spsLen);
      j += spsLen;

      // PPS
      x264_nal_t& ppsNal = nalOut[++i];  // the PPS always comes after the SPS
      unsigned char* pps = ppsNal.p_payload + 4;
      int ppsLen = ppsNal.i_payload - 4;
      pkt[j++] = 0x01;
      pkt[j++] = (ppsLen >> 8) & 0xff;  // ppsLen should always be 4 bytes
      pkt[j++] = (ppsLen)&0xff;
      memcpy(&pkt[j], pps, ppsLen);
      j += ppsLen;

      VideoStitch::IO::DataPacket _pkt(storage, j);
      _pkt.timestamp = uint64_t(picOut.i_dts);
      _pkt.type = VideoStitch::IO::PacketType_VideoSPS;
      packets.push_back(_pkt);
    }
  }
  return true;
}

char* X264Encoder::metadata(char* enc, char* pend) {
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videocodecid, 7.);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videodatarate, bitrateMax);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_framerate, framerate.num / (double)framerate.den);
  return enc;
}

int X264Encoder::getBitRate() const {
  if (paramData.rc.i_vbv_max_bitrate)
    return paramData.rc.i_vbv_max_bitrate;
  else
    return paramData.rc.i_bitrate;
}

bool X264Encoder::setBitRate(uint32_t maxBitrate, uint32_t bufferSize) {
  setBitRateParams(maxBitrate, bufferSize);

  int retVal = x264_encoder_reconfig(x264, &paramData);
  if (retVal < 0) {
    Logger::get(Logger::Error) << "RTMP : Could not set new encoder bitrate, error value " << retVal << std::endl;
    return false;
  }
  Logger::info("RTMP") << "Set bit rate to " << maxBitrate << std::endl;
  return true;
}

void X264Encoder::setBitRateParams(uint32_t maxBitrate, uint32_t bufferSize) {
  if (useCBR) {
    paramData.rc.i_bitrate = maxBitrate;
  } else if (maxBitrate != uint32_t(-1)) {
    paramData.rc.i_vbv_max_bitrate = maxBitrate;  // vbv-maxrate
    paramData.rc.i_vbv_buffer_size = maxBitrate;  // vbv-bufsize default if not set
  }
  if (bufferSize != uint32_t(-1)) {
    paramData.rc.i_vbv_buffer_size = bufferSize;  // vbv-bufsize
  }
}

Potential<VideoEncoder> X264Encoder::createX264Encoder(const Ptv::Value& config, int width, int height,
                                                       FrameRate framerate) {
  INT(config, bitrate, 2000);
  INT(config, buffer_size, -1);
  STRING(config, preset, "medium");
  STRING(config, tune, "");
  STRING(config, profile, "baseline");
  STRING(config, level, "3.1");
  STRING(config, bitrate_mode, "VBR");
  INT(config, quality_balance, 20);
  INT(config, gop, 0);
  INT(config, b_frames, 2);
  BOOLE(config, cbr_padding, true);

  VideoStitch::IO::ColorDescription colorDesc;
  colorDesc.fullRange = false;
  colorDesc.primaries = VideoStitch::IO::ColorPrimaries_BT709;
  colorDesc.transfer = VideoStitch::IO::ColorTransfer_IEC6196621;
  colorDesc.matrix =
      width >= 1280 || height > 576 ? VideoStitch::IO::ColorMatrix_BT709 : VideoStitch::IO::ColorMatrix_SMPTE170M;
  return Potential<VideoEncoder>(new X264Encoder(framerate, width, height, preset, tune, profile, level, bitrate_mode,
                                                 quality_balance, colorDesc, bitrate, buffer_size, cbr_padding, gop,
                                                 b_frames));
}

}  // namespace Output
}  // namespace VideoStitch
