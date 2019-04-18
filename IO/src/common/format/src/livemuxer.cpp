// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "livemuxer.hpp"

#include "libvideostitch/logging.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

#include <iostream>

namespace VideoStitch {
namespace Output {

LiveMuxer::LiveMuxer(size_t index, const std::string& /* url */, std::vector<AVEncoder>& codecs)
    : Muxer(index,
            "flv",  // RTMP requires the 'flv' container
            codecs, nullptr) {}

bool LiveMuxer::openResource(const std::string& url) {
  writerGlobalHeaders();
  if (avformat_network_init() != 0) {
    Logger::get(Logger::Error) << "[libavoutput] could initialize network connection for: " << url
                               << ", disable output." << std::endl;
    return false;
  }
  int r = avio_open(&formatCtx->pb, url.c_str(), AVIO_FLAG_WRITE);
  if (r < 0) {
    Logger::get(Logger::Error) << "[libavoutput] could not open " << url << ", disable output. Error : " << r << " : "
                               << Util::errorString(r) << std::endl;
    return false;
  }
  return true;
}

LiveMuxer::~LiveMuxer() { avformat_network_deinit(); }

void LiveMuxer::writeTrailer() {
  if (formatCtx != nullptr) {
    int r = avio_close(formatCtx->pb);
    if (r < 0) {
      Logger::get(Logger::Error) << "[libavoutput] Could not close livemuxer.  Error : " << r << " : "
                                 << Util::errorString(r) << std::endl;
    }
  }
}

}  // namespace Output
}  // namespace VideoStitch
