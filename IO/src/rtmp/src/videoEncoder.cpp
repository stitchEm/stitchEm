// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <sstream>
#include <algorithm>
#include "videoEncoder.hpp"
#include "mockEncoder.hpp"
#include "x264Encoder.hpp"

#ifdef SUP_QUICKSYNC
#include "qsvEncoder.hpp"
#endif

#ifdef SUP_NVENC
#include "nvenc.hpp"
#elif defined(SUP_NVENC_M)
#include "NvV4l2Encoder.hpp"
#endif

const static uint8_t start_seq[] = {0, 0, 1};

namespace VideoStitch {
namespace Output {

Potential<VideoEncoder> VideoEncoder::createVideoEncoder(const Ptv::Value &config, int width, int height,
                                                         FrameRate framerate, const std::string &encoderType) {
  if (("x264" == encoderType) || ("h264" == encoderType)) {
    return X264Encoder::createX264Encoder(config, width, height, framerate);
  } else if ("mock" == encoderType) {
    return MockEncoder::createMockEncoder();
  }
#ifdef SUP_QUICKSYNC
  else if (encoderType.find("qsv") != std::string::npos) {
    return QSVEncoder::createQSVEncoder(config, width, height, framerate);
  }
#endif
#ifdef SUP_NVENC
  else if (encoderType.find("nvenc") != std::string::npos) {
    return NvEncoder::createNvEncoder(config, width, height, framerate);
  }
#elif defined(SUP_NVENC_M)
  else if (encoderType.find("nvenc") != std::string::npos) {
    return NvV4l2Encoder::createNvV4l2Encoder(config, width, height, framerate);
  }
#endif
  std::stringstream msg;
  msg << "[RTMP] Unknown encoderType: " << encoderType;
  return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, msg.str());
}

void VideoEncoder::supportedCodecs(std::vector<std::string> &codecs) {
  codecs.push_back("h264");
#ifdef SUP_NVENC
  NvEncoder::supportedEncoders(codecs);
#elif defined(SUP_NVENC_M)
  NvV4l2Encoder::supportedEncoders(codecs);
#endif
#ifdef SUP_QUICKSYNC
  QSVEncoder::supportedEncoders(codecs);
#endif
}

void VideoEncoder::createDataPacket(std::vector<x264_nal_t> &nalOut, std::vector<VideoStitch::IO::DataPacket> &packets,
                                    mtime_t pts, mtime_t dts) {
  // NALU into RTMP packets
  // https://en.wikipedia.org/wiki/Network_Abstraction_Layer
  for (size_t i = 0; i < nalOut.size(); ++i) {
    x264_nal_t &nal = nalOut[i];

    if (nal.i_type == NAL_SLICE_IDR || nal.i_type == NAL_SLICE || nal.i_type == NAL_AUD || nal.i_type == NAL_SEI) {
      int j = 0;
      VideoStitch::IO::DataPacket pkt(nal.i_payload + 9);
      pkt.timestamp = dts;

      // NAL unit metadata
      pkt[j++] = (nal.i_type == NAL_SLICE_IDR) ? 0x17 : 0x27;  // keyframe vs. I/P/B slice
      // NAL unit delimiter prefix
      pkt[j++] = 0x01;
      // Composition time : See ISO 14496-12, 8.15.3
      // offset between decoding time and presentation/composition time
      uint64_t composition_time_offset = pts - pkt.timestamp;
      pkt[j++] = composition_time_offset >> 16 & 0xff;
      pkt[j++] = composition_time_offset >> 8 & 0xff;
      pkt[j++] = composition_time_offset & 0xff;
      // NAL unit size
      pkt[j++] = nal.i_payload >> 24 & 0xff;
      pkt[j++] = nal.i_payload >> 16 & 0xff;
      pkt[j++] = nal.i_payload >> 8 & 0xff;
      pkt[j++] = nal.i_payload & 0xff;
      // NAL unit data
      memcpy(&pkt[j], nal.p_payload, nal.i_payload);
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
      unsigned char *sps = nal.p_payload;
      int spsLen = nal.i_payload;
      // DEBUG
      /*
      int width, height, fps;
      h264_decode_sps(sps, spsLen, width, height, fps);
      */

      int j = 0;

      VideoStitch::IO::DataPacket::Storage storage = VideoStitch::IO::DataPacket::make_storage(1024);
      unsigned char *pkt = storage.get();

      pkt[j++] = 0x17;  // H.264 K frame
      pkt[j++] = 0x00;  // codec config packet

      // start time on 24 bits
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;

      // ISO/IEC 14496-15 AVCDecoderConfigurationRecord
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
      x264_nal_t &ppsNal = nalOut[++i];  // the PPS always comes after the SPS
      unsigned char *pps = ppsNal.p_payload;
      int ppsLen = ppsNal.i_payload;
      pkt[j++] = 0x01;
      pkt[j++] = (ppsLen >> 8) & 0xff;  // ppsLen should always be 4 bytes
      pkt[j++] = (ppsLen)&0xff;
      memcpy(&pkt[j], pps, ppsLen);
      j += ppsLen;

      VideoStitch::IO::DataPacket _pkt(storage, j);
      _pkt.timestamp = dts;
      _pkt.type = VideoStitch::IO::PacketType_VideoSPS;
      packets.push_back(_pkt);
    }
  }
}

void VideoEncoder::createHEVCPacket(std::vector<x265_nal> &nalOut, std::vector<VideoStitch::IO::DataPacket> &packets,
                                    mtime_t pts, mtime_t dts) {
  // NALU into RTMP packets
  // https://en.wikipedia.org/wiki/Network_Abstraction_Layer
  for (size_t i = 0; i < nalOut.size(); ++i) {
    x265_nal &nal = nalOut[i];

    if (/*nal.type >= NAL_UNIT_CODED_SLICE_TRAIL_N &&*/ nal.type <= NAL_UNIT_CODED_SLICE_CRA) {
      int j = 0;
      VideoStitch::IO::DataPacket pkt(nal.sizeBytes + 9);
      pkt.timestamp = dts;

      // NAL unit metadata
      pkt[j++] = (nal.type == NAL_UNIT_CODED_SLICE_IDR_W_RADL || nal.type == NAL_UNIT_CODED_SLICE_IDR_N_LP)
                     ? 0x18
                     : 0x28;  // keyframe vs. I/P/B slice
      // NAL unit delimiter prefix
      pkt[j++] = 0x01;
      // Composition time : See ISO 14496-12, 8.15.3
      // offset between decoding time and presentation/composition time
      uint64_t composition_time_offset = pts - pkt.timestamp;
      pkt[j++] = composition_time_offset >> 16 & 0xff;
      pkt[j++] = composition_time_offset >> 8 & 0xff;
      pkt[j++] = composition_time_offset & 0xff;
      // NAL unit size
      pkt[j++] = nal.sizeBytes >> 24 & 0xff;
      pkt[j++] = nal.sizeBytes >> 16 & 0xff;
      pkt[j++] = nal.sizeBytes >> 8 & 0xff;
      pkt[j++] = nal.sizeBytes & 0xff;
      // NAL unit data
      memcpy(&pkt[j], nal.payload, nal.sizeBytes);
      switch (nal.type) {
        /* Trailing pictures */
        case NAL_UNIT_CODED_SLICE_TRAIL_N:
          pkt.type = VideoStitch::IO::PacketType_VideoDisposable;
          break;
        case NAL_UNIT_CODED_SLICE_TRAIL_R:
          pkt.type = VideoStitch::IO::PacketType_VideoLow;
          break;
        /* Leading pictures */
        case NAL_UNIT_CODED_SLICE_RADL_N:
        case NAL_UNIT_CODED_SLICE_RADL_R:
        case NAL_UNIT_CODED_SLICE_RASL_N:
        case NAL_UNIT_CODED_SLICE_RASL_R:
          pkt.type = VideoStitch::IO::PacketType_VideoHigh;
          break;
        /* IDR And other Intra Random Access Pictures */
        case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
        case NAL_UNIT_CODED_SLICE_IDR_N_LP:
        default:
          pkt.type = VideoStitch::IO::PacketType_VideoHighest;
          break;
      }

      packets.push_back(pkt);
    } else if (nal.type == NAL_UNIT_VPS) {
      x265_nal &vpsNal = nalOut[i++];
      unsigned char *vps = vpsNal.payload;
      int vpsLen = vpsNal.sizeBytes;
      x265_nal &spsNal = nalOut[i++];  // the SPS always comes after the VPS
      unsigned char *sps = spsNal.payload;
      int spsLen = spsNal.sizeBytes;
      x265_nal &ppsNal = nalOut[i];  // the PPS always comes after the SPS
      unsigned char *pps = ppsNal.payload;
      int ppsLen = ppsNal.sizeBytes;

      // ISO/IEC 14496-15 HEVC decoder configuration records
      // 8.3.3.1.2 Syntax

      int j = 0;

      VideoStitch::IO::DataPacket::Storage storage = VideoStitch::IO::DataPacket::make_storage(1024);
      unsigned char *pkt = storage.get();

      // FLV header
      pkt[j++] = 0x18;  // HEVC Video Packet
      pkt[j++] = 0x00;  // codec config packet
      // start time on 24 bits
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;
      pkt[j++] = 0x00;

      // Most of this information is redundant with the sequence parameter set,
      // see ISO/IEC 14496-15 7.3.2.2

      pkt[j++] = 0x01;  // configurationVersion

      // --- PTL (profile-tier-level) ----
      // the next twelve bytes are the general profile space, tier flag,
      // profile idc, profile compatibility slags,
      // constraints indicator flags, and level idc
      for (int i = 1; i <= 12; ++i) {
        pkt[j++] = sps[i];
      }

      // min_spatial_segmentation_idc (12), front padded with 1s
      // this resides in the VUI of the SPS, currently not parsed
      // VUI parameters are not required for constructing the luma or chroma samples by the decoding process.
      // Set everything to 0.
      pkt[j++] = 0xf0;
      pkt[j++] = 0x0;
      // parallelismType (2), front padded with 1s, set to 0 too
      pkt[j++] = 0xfc;

      // ---- colorspace indication ----
      // chromat format idc (2), front padded with 1s
      // 0 means monochrome, 1 is 4:2:0, 2 is 4:2:2, 3 is 4:4:4
      pkt[j++] = 0xfc | 0x1;  // hardcode 4:2:0, for now it's always NV12
      // bit depth luma minus 8 (3), front padded with 1s
      pkt[j++] = 0x0;  // for now always 8 bits
      // bit depth chroma minus 8 (3), front padded with 1s
      pkt[j++] = 0x0;  // for now always 8 bits

      // ---- framerate indication ----
      // avg frame rate
      // It's unclear how to properly compute these fields, so
      // let's always set them to values meaning 'unspecified'.
      pkt[j++] = 0;
      pkt[j++] = 0;
      // constantFrameRate (2) : 0 for unspecified,
      // numTemporalLayers (3), and temporalIdNested (1) : the four least significant bits of the first byte of the SPS
      // lengthSizeMinusOne (2) : 3, since our length is encoded on 4 bytes on the slices NAL units
      pkt[j++] = (unsigned char)((sps[0] & 0xf) << 2) | 0x3;

      // num of arrays (VPS, SPS, PPS)
      pkt[j++] = 0x03;

      // ----- VPS
      pkt[j++] = NAL_UNIT_VPS;  // array completeness (1) : 0 here, bit (1) reserved = 0, type (ISO/IEC 23008-2)
      pkt[j++] = 0x0;           // number of NAL units (16 bits)
      pkt[j++] = 0x1;
      pkt[j++] = (vpsLen >> 8) & 0xff;
      pkt[j++] = vpsLen & 0xff;
      memcpy(&pkt[j], vps, vpsLen);
      j += vpsLen;

      // ----- SPS
      pkt[j++] = NAL_UNIT_SPS;  // array completeness (1) : 0 here, bit (1) reserved = 0, type (ISO/IEC 23008-2)
      pkt[j++] = 0x0;           // number of NAL units (16 bits)
      pkt[j++] = 0x1;
      pkt[j++] = (spsLen >> 8) & 0xff;
      pkt[j++] = spsLen & 0xff;
      memcpy(&pkt[j], sps, spsLen);
      j += spsLen;

      // ----- PPS
      pkt[j++] = NAL_UNIT_PPS;  // array completeness (1) : 0 here, bit (1) reserved = 0, type (ISO/IEC 23008-2)
      pkt[j++] = 0x0;           // number of NAL units (16 bits)
      pkt[j++] = 0x1;
      pkt[j++] = (ppsLen >> 8) & 0xff;  // ppsLen should always be 4 bytes
      pkt[j++] = (ppsLen)&0xff;
      memcpy(&pkt[j], pps, ppsLen);
      j += ppsLen;

      VideoStitch::IO::DataPacket _pkt(storage, j);
      _pkt.timestamp = dts;
      _pkt.type = VideoStitch::IO::PacketType_VideoSPS;
      packets.push_back(_pkt);
    }
  }
}

void VideoEncoder::header(std::vector<VideoStitch::IO::DataPacket> &packets) {
  /* create specific nal for RTMP configuration  */
  if (headPkt.size()) {
    std::vector<x264_nal_t> nalOut;
    x264_nal_t nal;

    uint8_t *start = (uint8_t *)headPkt.data();
    uint8_t *end = start + int(headPkt.size());
    start = std::search(start, end, start_seq, start_seq + 3);
    while (start != end) {
      decltype(start) next = std::search(start + 1, end, start_seq, start_seq + 3);
      nal.i_ref_idc = start[3] >> 5;
      nal.i_type = start[3] & 0x1f;
      nal.p_payload = start + 3;
      nal.i_payload = int(next - start - 3);
      nalOut.push_back(nal);
      start = next;
    }

    VideoEncoder::createDataPacket(nalOut, packets, 0, 0);
  }
}
}  // namespace Output
}  // namespace VideoStitch
