// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoEncoder.hpp"

#include <memory>

#define BUFFER_SIZE 32768
#define GOT_A_NAL_CROSS_BUFFER BUFFER_SIZE + 1
#define GOT_A_NAL_INCLUDE_A_BUFFER BUFFER_SIZE + 2
#define NO_MORE_BUFFER_TO_READ BUFFER_SIZE + 3

namespace VideoStitch {
namespace Output {

struct NaluUnit {
  int type;
  int size;
  unsigned char *data;
};

/**
 * The MockEncoder takes an Annex-B H264 bytesteam, and at each call to encode(),
 * provides the Publisher with the next encoded frame in the bytestream.
 *
 * You can generate an Annex-B bytestream from an MP4 container by extracting the
 * h.264 NAL units from it with FFMpeg:
 * > ffmpeg -i my_movie.mp4 -vcodec copy -vbsf h264_mp4toannexb -an of.h264
 *
 */

class MockEncoder : public VideoEncoder {
 public:
  MockEncoder();
  ~MockEncoder();

  static Potential<VideoEncoder> createMockEncoder();

  bool encode(const Frame &, std::vector<VideoStitch::IO::DataPacket> &packets);
  void getHeaders(VideoStitch::IO::DataPacket &packet);

  char *metadata(char *enc, char *) { return enc; }
  int getBitRate() const { return 1000; }
  bool dynamicBitrateSupported() const { return false; }
  bool setBitRate(uint32_t /*maxBitrate*/, uint32_t /*bufferSize*/) { return false; }

 private:
  static int read_buffer(uint8_t *buf, int buf_size);

  int readFirstNaluFromBuf(NaluUnit &nalu);
  int readOneNaluFromBuf(NaluUnit &nalu, int (*read_buffer)(uint8_t *buf, int buf_size));

  unsigned int nalhead_pos;
  unsigned char *m_pFileBuf;
  unsigned int m_nFileBufSize;
  unsigned char *m_pFileBuf_tmp;
  unsigned char *m_pFileBuf_tmp_old;  // used for realloc

  static FILE *fp_send;
};

}  // namespace Output
}  // namespace VideoStitch
