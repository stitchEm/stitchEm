// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "filemuxer.hpp"
#include "libvideostitch/logging.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

#include <iostream>
#include <sstream>
#include <cstdio>

#include "qt-faststart.hpp"

static const char FILE_TMP_SUFFIX[] = ".vs.tmp";

namespace VideoStitch {
namespace Output {

FileMuxer::FileMuxer(size_t index, const std::string &format, const std::string &, std::vector<AVEncoder> &codecs,
                     const AVDictionary *config)
    : Muxer(index, format, codecs, config) {
  /* config will be modified when used by avformat_write_header() so we get relevant info before */
  reserved_moov = !!av_dict_get(config, "moov_size", nullptr, 0);
}
FileMuxer::~FileMuxer() {
  if (formatCtx && getMuxedSize() == 0) {
    std::remove(formatCtx->filename);
  }
}

void FileMuxer::writeTrailer() {
  Muxer::writeTrailer();
  std::string filename = std::string(formatCtx->filename);
  if (!strcmp(formatCtx->oformat->name, "mp4")) {
    // optimize for web: write moov before mdat
    if (!MP4WebOptimizerInternal(filename)) {
      Logger::get(Logger::Warning) << "[libavoutput] couldn't optimize the mp4 for the Web." << std::endl;
    }
  }
  // x264 with libav leaves a trace file in the working directory
  std::remove("x264_2pass.log");
  std::remove("x264_2pass.log.temp");
  std::remove("x264_2pass.log.mbtree");
}

bool FileMuxer::openResource(const std::string &filename) {
  writerGlobalHeaders();
  // TODO: use better file manipulation for this class
  // get the first extension recommended by libav
  std::ostringstream fn;
  fn << filename;
  std::istringstream formatExts(formatCtx->oformat->extensions);
  std::string filenameExt;
  std::getline(formatExts, filenameExt, ',');

  if ((filenameExt == "mp4") && (!reserved_moov)) {
    fn << "." << filenameExt << FILE_TMP_SUFFIX;
  } else {
    fn << "." << filenameExt;
  }

  const std::string completeName = fn.str();
  // open the output file, if needed
  if (!(formatCtx->flags & AVFMT_NOFILE)) {
    int r = avio_open(&formatCtx->pb, completeName.c_str(), AVIO_FLAG_READ_WRITE);
    if (r < 0) {
      Logger::get(Logger::Error) << "[libavoutput] could not open " << filename << ", disable output. Error : " << r
                                 << " : " << Util::errorString(r) << std::endl;
      return false;
    }
    strncpy(formatCtx->filename, completeName.c_str(), sizeof(formatCtx->filename));
  }
  return true;
}

bool FileMuxer::MP4WebOptimizerInternal(const std::string &srcFile) {
  // TODO: use better file manipulation for this class
  int e = 0;

  uint32_t nbChannels = 0;
  for (unsigned i = 0; i < formatCtx->nb_streams; i++) {
    if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      nbChannels = formatCtx->streams[i]->codecpar->channels;
      break;
    }
  }

  if (reserved_moov) {
    bool res = qt_faststart(srcFile.c_str(), srcFile.c_str(), nbChannels);
    if (!res) {
      Logger::get(Logger::Warning) << "[libavoutput] Error manipulating output file." << std::endl;
      return false;
    }
    return res;
  }

  const std::string dstFile = srcFile.substr(0, srcFile.size() - strlen(FILE_TMP_SUFFIX));
  bool res = qt_faststart(srcFile.c_str(), dstFile.c_str(), nbChannels);
  if (res) {
    e = std::remove(srcFile.c_str());
  } else {
    e = std::remove(dstFile.c_str());
    if (!e) {
      e = std::rename(srcFile.c_str(), dstFile.c_str());
    }
  }
  if (e) {
    Logger::get(Logger::Warning) << "[libavoutput] Error manipulating output file." << std::endl;
    return false;
  } else {
    return res;
  }
}

}  // namespace Output
}  // namespace VideoStitch
