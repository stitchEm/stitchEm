// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "librtmpIncludes.hpp"
#include "mockEncoder.hpp"

#include "libvideostitch/logging.hpp"

#include <cstring>
#include <cmath>
#include <stdlib.h>

namespace VideoStitch {
namespace Output {

FILE *MockEncoder::fp_send;

MockEncoder::MockEncoder() {
  fp_send = fopen("C:\\Users\\nlz\\Videos\\edhec.h264", "rb");
  nalhead_pos = 0;
  m_nFileBufSize = ENC_BUFFER_SIZE;
  m_pFileBuf = (unsigned char *)malloc(ENC_BUFFER_SIZE);
  m_pFileBuf_tmp = (unsigned char *)malloc(ENC_BUFFER_SIZE);
  std::memset(m_pFileBuf, 0, ENC_BUFFER_SIZE);
  read_buffer(m_pFileBuf, m_nFileBufSize);
}

bool MockEncoder::encode(const Frame &frame, std::vector<VideoStitch::IO::DataPacket> &packets) {
  NaluUnit naluUnit;
  naluUnit.data = nullptr;
  naluUnit.size = 0;
  naluUnit.type = 0x00;

ignore_nal:
  if (!readOneNaluFromBuf(naluUnit, read_buffer)) return false;
  Logger::get(Logger::Error) << "RTMP : Mock NALU type " << naluUnit.type << " size=" << naluUnit.size << std::endl;

  if (naluUnit.type > 0x05)  // non-VCLs NAL (Video Coding Layer)
    goto ignore_nal;

  VideoStitch::IO::DataPacket pkt(naluUnit.size + 9);
  pkt.timestamp = mtime_t(std::round(frame.pts / 1000.0));
  int j = 0;
  // NAL unit metadata
  if (naluUnit.type == 0x05) {
    pkt[j++] = 0x17;
    pkt.type = VideoStitch::IO::PacketType_VideoHighest;
    // send the SPS/PPS before each keyframe
    VideoStitch::IO::DataPacket spsPps(headPkt);
    spsPps.type = VideoStitch::IO::PacketType_VideoSPS;
    spsPps.timestamp = pkt.timestamp;
    packets.push_back(spsPps);
  } else {
    pkt[j++] = 0x27;
    pkt.type = VideoStitch::IO::PacketType_VideoHigh;
  }
  // NAL unit delimiter prefix
  pkt[j++] = 0x01;
  pkt[j++] = 0x00;
  pkt[j++] = 0x00;
  pkt[j++] = 0x00;
  // NAL unit size
  pkt[j++] = naluUnit.size >> 24 & 0xff;
  pkt[j++] = naluUnit.size >> 16 & 0xff;
  pkt[j++] = naluUnit.size >> 8 & 0xff;
  pkt[j++] = naluUnit.size & 0xff;
  // NAL unit data
  std::memcpy(&pkt[j], naluUnit.data, naluUnit.size);

  packets.push_back(pkt);
  return true;
}

void MockEncoder::getHeaders(VideoStitch::IO::DataPacket &packet) {
  if (!headPkt.size()) {
    NaluUnit naluUnit;
    naluUnit.data = nullptr;
    naluUnit.size = 0;
    naluUnit.type = 0x00;

    readFirstNaluFromBuf(naluUnit);
    while (naluUnit.type != 0x07) {
      readOneNaluFromBuf(naluUnit, read_buffer);
    }
    int j = 0;
    headPkt.resize(1024);
    std::fill(headPkt.begin(), headPkt.end(), 0);

    // see documentation in X264Encoder
    headPkt[j++] = 0x17;
    headPkt[j++] = 0x00;
    headPkt[j++] = 0x00;
    headPkt[j++] = 0x00;
    headPkt[j++] = 0x00;
    headPkt[j++] = 0x01;
    headPkt[j++] = naluUnit.data[1];
    headPkt[j++] = naluUnit.data[2];
    headPkt[j++] = naluUnit.data[3];
    headPkt[j++] = 0xff;
    headPkt[j++] = 0xe1;
    headPkt[j++] = (naluUnit.size >> 8) & 0xff;
    headPkt[j++] = naluUnit.size & 0xff;
    std::memcpy(&headPkt[j], naluUnit.data, naluUnit.size);
    j += naluUnit.size;
    readOneNaluFromBuf(naluUnit, read_buffer);
    headPkt[j++] = 0x01;
    headPkt[j++] = (naluUnit.size >> 8) & 0xff;
    headPkt[j++] = (naluUnit.size) & 0xff;
    std::memcpy(&headPkt[j], naluUnit.data, naluUnit.size);
  }
  packet = headPkt;
}

int MockEncoder::read_buffer(uint8_t *buf, int buf_size) {
  if (!feof(fp_send)) {
    size_t true_size = fread(buf, 1, buf_size, fp_send);
    return (int)true_size;
  } else {
    return -1;
  }
}

int MockEncoder::readFirstNaluFromBuf(NaluUnit &nalu) {
  unsigned int naltail_pos;
  std::memset(m_pFileBuf_tmp, 0, ENC_BUFFER_SIZE);
  while (nalhead_pos < m_nFileBufSize) {
    // search for nal header
    if (m_pFileBuf[nalhead_pos++] == 0x00 && m_pFileBuf[nalhead_pos++] == 0x00) {
      if (m_pFileBuf[nalhead_pos++] == 0x01)
        goto gotnal_head;
      else {
        // cuz we have done an i++ before,so we need to roll back now
        nalhead_pos--;
        if (m_pFileBuf[nalhead_pos++] == 0x00 && m_pFileBuf[nalhead_pos++] == 0x01)
          goto gotnal_head;
        else
          continue;
      }
    } else
      continue;

    // search for nal tail which is also the head of next nal
  gotnal_head:
    // normal case:the whole nal is in this m_pFileBuf
    naltail_pos = nalhead_pos;
    while (naltail_pos < m_nFileBufSize) {
      if (m_pFileBuf[naltail_pos++] == 0x00 && m_pFileBuf[naltail_pos++] == 0x00) {
        if (m_pFileBuf[naltail_pos++] == 0x01) {
          nalu.size = (naltail_pos - 3) - nalhead_pos;
          break;
        } else {
          naltail_pos--;
          if (m_pFileBuf[naltail_pos++] == 0x00 && m_pFileBuf[naltail_pos++] == 0x01) {
            nalu.size = (naltail_pos - 4) - nalhead_pos;
            break;
          }
        }
      }
    }

    nalu.type = m_pFileBuf[nalhead_pos] & 0x1f;
    std::memcpy(m_pFileBuf_tmp, m_pFileBuf + nalhead_pos, nalu.size);
    nalu.data = m_pFileBuf_tmp;
    nalhead_pos = naltail_pos;
    return TRUE;
  }
  return FALSE;
}

int MockEncoder::readOneNaluFromBuf(NaluUnit &nalu, int (*read_buffer)(uint8_t *buf, int buf_size)) {
  unsigned int naltail_pos = nalhead_pos;
  int ret;
  int nalustart;
  std::memset(m_pFileBuf_tmp, 0, ENC_BUFFER_SIZE);
  nalu.size = 0;
  for (;;) {
    if (nalhead_pos == NO_MORE_BUFFER_TO_READ) return FALSE;
    while (naltail_pos < m_nFileBufSize) {
      // search for nal tail
      if (m_pFileBuf[naltail_pos++] == 0x00 && m_pFileBuf[naltail_pos++] == 0x00) {
        if (m_pFileBuf[naltail_pos++] == 0x01) {
          nalustart = 3;
          goto gotnal;
        } else {
          naltail_pos--;
          if (m_pFileBuf[naltail_pos++] == 0x00 && m_pFileBuf[naltail_pos++] == 0x01) {
            nalustart = 4;
            goto gotnal;
          } else
            continue;
        }
      } else
        continue;

    gotnal:
      // special case1 : parts of the nal lies in a m_pFileBuf and we have to read from buffer
      // again to get the rest part of this nal
      if (nalhead_pos == GOT_A_NAL_CROSS_BUFFER || nalhead_pos == GOT_A_NAL_INCLUDE_A_BUFFER) {
        nalu.size = nalu.size + naltail_pos - nalustart;
        if (nalu.size > ENC_BUFFER_SIZE) {
          m_pFileBuf_tmp_old = m_pFileBuf_tmp;  //// save pointer in case realloc fails
          if ((m_pFileBuf_tmp = (unsigned char *)realloc(m_pFileBuf_tmp, nalu.size)) == NULL) {
            free(m_pFileBuf_tmp_old);  // free original block
            return FALSE;
          }
        }
        std::memcpy(m_pFileBuf_tmp + nalu.size + nalustart - naltail_pos, m_pFileBuf, naltail_pos - nalustart);
        nalu.data = m_pFileBuf_tmp;
        nalhead_pos = naltail_pos;
        return TRUE;
      }
      // normal case:the whole nal is in this m_pFileBuf
      else {
        nalu.type = m_pFileBuf[nalhead_pos] & 0x1f;
        nalu.size = naltail_pos - nalhead_pos - nalustart;
        if (nalu.type == 0x06) {
          nalhead_pos = naltail_pos;
          continue;
        }
        std::memcpy(m_pFileBuf_tmp, m_pFileBuf + nalhead_pos, nalu.size);
        nalu.data = m_pFileBuf_tmp;
        nalhead_pos = naltail_pos;
        return TRUE;
      }
    }

    if (naltail_pos >= m_nFileBufSize && nalhead_pos != GOT_A_NAL_CROSS_BUFFER &&
        nalhead_pos != GOT_A_NAL_INCLUDE_A_BUFFER) {
      nalu.size = ENC_BUFFER_SIZE - nalhead_pos;
      nalu.type = m_pFileBuf[nalhead_pos] & 0x1f;
      std::memcpy(m_pFileBuf_tmp, m_pFileBuf + nalhead_pos, nalu.size);
      if ((ret = read_buffer(m_pFileBuf, m_nFileBufSize)) < ENC_BUFFER_SIZE) {
        std::memcpy(m_pFileBuf_tmp + nalu.size, m_pFileBuf, ret);
        nalu.size = nalu.size + ret;
        nalu.data = m_pFileBuf_tmp;
        nalhead_pos = NO_MORE_BUFFER_TO_READ;
        return FALSE;
      }
      naltail_pos = 0;
      nalhead_pos = GOT_A_NAL_CROSS_BUFFER;
      continue;
    }
    if (nalhead_pos == GOT_A_NAL_CROSS_BUFFER || nalhead_pos == GOT_A_NAL_INCLUDE_A_BUFFER) {
      nalu.size = ENC_BUFFER_SIZE + nalu.size;

      m_pFileBuf_tmp_old = m_pFileBuf_tmp;  //// save pointer in case realloc fails
      if ((m_pFileBuf_tmp = (unsigned char *)realloc(m_pFileBuf_tmp, nalu.size)) == NULL) {
        free(m_pFileBuf_tmp_old);  // free original block
        return FALSE;
      }

      std::memcpy(m_pFileBuf_tmp + nalu.size - ENC_BUFFER_SIZE, m_pFileBuf, ENC_BUFFER_SIZE);

      if ((ret = read_buffer(m_pFileBuf, m_nFileBufSize)) < ENC_BUFFER_SIZE) {
        std::memcpy(m_pFileBuf_tmp + nalu.size, m_pFileBuf, ret);
        nalu.size = nalu.size + ret;
        nalu.data = m_pFileBuf_tmp;
        nalhead_pos = NO_MORE_BUFFER_TO_READ;
        return FALSE;
      }
      naltail_pos = 0;
      nalhead_pos = GOT_A_NAL_INCLUDE_A_BUFFER;
      continue;
    }
  }
}

Potential<VideoEncoder> MockEncoder::createMockEncoder() { return Potential<VideoEncoder>(new MockEncoder); }

MockEncoder::~MockEncoder() {
  if (fp_send) {
    fclose(fp_send);
  }
}

}  // namespace Output
}  // namespace VideoStitch
