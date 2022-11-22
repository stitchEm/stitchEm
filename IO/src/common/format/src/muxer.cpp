// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "muxer.hpp"
#include "util.hpp"

#include "libvideostitch/logging.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
}

#ifdef _WIN32
#include <windows.h>

// FFMPEG negates system error codes
#define FFMPEG_SOCKET_TIMEOUT_ERROR (-WSAECONNRESET)
#else
#include <errno.h>

// FFMPEG negates system error codes
#define FFMPEG_SOCKET_TIMEOUT_ERROR (-ETIMEDOUT)
#endif

#include <iostream>

static std::string AVtag("libavoutput");

namespace VideoStitch {
void PacketQueue::pushPacket(std::shared_ptr<struct AVPacket> pkt, const int streamId) {
  if (!streams[streamId]) {
    Logger::warning(AVtag) << "No stream available for id : " << streamId << std::endl;
    return;
  }
  pkt->stream_index = streams[streamId]->index;
  {
    std::lock_guard<std::mutex> lock(mutex);
    packets.push(pkt);
  }
  cond.notify_one();
}

void PacketQueue::rescaleTimestamp(AVCodecContext* codecCtx, AVStream* stream, AVPacket* pkt) {
  if (pkt->pts != (int64_t)AV_NOPTS_VALUE) {
    pkt->dts = av_rescale_q(pkt->dts, codecCtx->time_base, stream->time_base);
    pkt->pts = av_rescale_q(pkt->pts, codecCtx->time_base, stream->time_base);
  }
  if (pkt->duration > 0) {
    pkt->duration = (int)av_rescale_q(pkt->duration, codecCtx->time_base, stream->time_base);
  }
}

std::shared_ptr<struct AVPacket> PacketQueue::popPacket() {
  if (packets.empty()) {
    return nullptr;
  }
  auto pkt = packets.front();
  packets.pop();
  return pkt;
}

bool PacketQueue::isEmpty() const { return packets.empty(); }

namespace Output {

Muxer::Muxer(size_t index, const std::string& format, std::vector<AVEncoder>& encoders, const AVDictionary* config)
    : formatCtx(avformat_alloc_context()),
      packets(encoders.size()),
      m_threadStatus(MuxerThreadStatus::OK),
      index(index) {
  // setup container
  AVOutputFormat* of = av_guess_format(format.c_str(), nullptr, nullptr);
  if (!of) {
    Logger::error(AVtag) << "Couldn't guess container from file extension" << std::endl;
    setThreadStatus(MuxerThreadStatus::CreateError);
    return;
  }

  if (!formatCtx) {
    Logger::error(AVtag) << "Format context couldn't be allocated." << std::endl;
    setThreadStatus(MuxerThreadStatus::CreateError);
    return;
  }
  formatCtx->oformat = of;

  for (auto encoder : encoders) {
    AVStream* stream = nullptr;
    if (encoder.codecContext) {
      stream = avformat_new_stream(formatCtx, NULL);
      if (!stream) {
        Logger::error(AVtag) << "Could not create the stream " << encoder.id << ", disable output." << std::endl;
        setThreadStatus(MuxerThreadStatus::CreateError);
        return;
      }
      stream->id = formatCtx->nb_streams - 1;

      if (avcodec_parameters_from_context(stream->codecpar, encoder.codecContext) < 0) {
        Logger::error(AVtag) << "Could not copy parameters from stream " << stream->id
                             << " codec parameters to encoder context, disable output." << std::endl;
        setThreadStatus(MuxerThreadStatus::CreateError);
        return;
      }
      // copy encoder pointer
      encoderContexts[stream->index] = encoder.codecContext;
    }
    packets.streams[encoder.id] = stream;
  }

  m_config = nullptr;
  if (config) {
    av_dict_copy(&m_config, config, 0);
  }
}

Muxer::~Muxer() {
  avformat_free_context(formatCtx);
  av_dict_free(&m_config);
  encoderContexts.clear();
}

void Muxer::start() {
  Logger::debug(AVtag) << "Start muxer " << index << std::endl;
  thread = std::thread(&Muxer::run, this);
}

void Muxer::join() {
  {
    std::lock_guard<std::mutex> lock(packets.mutex);
    packets.shutDown = true;
  }
  packets.cond.notify_one();
  thread.join();
}

void Muxer::run() {
  writeHeader();
  while (true) {
    std::unique_lock<std::mutex> lock(packets.mutex);
    packets.cond.wait(lock, [this] { return !packets.isEmpty() || packets.shutDown; });

    if (packets.shutDown && packets.isEmpty()) {
      break;
    }

    auto pkt = packets.popPacket();
    if (pkt != nullptr) {
      writeFrame(pkt.get());
    }
  }
}

void Muxer::writerGlobalHeaders() {
  // Add codec metadata at the beginning and not on every keyframe
  if (formatCtx->oformat->flags & AVFMT_GLOBALHEADER) {
    for (auto& encoderCtx : encoderContexts) {
      if (encoderCtx.second) {
        encoderCtx.second->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      }
    }
  }
}

void Muxer::writeHeader() {
  const int r = avformat_write_header(formatCtx, &m_config);
  if (r < 0) {
    Logger::error(AVtag) << "Fatal error: can't write the container header for muxer " << index
                         << " . Error : " << Util::errorString(r) << std::endl;
    for (unsigned int i = 0; i < formatCtx->nb_streams; ++i) {
      if (formatCtx->streams[i]->codecpar) {
        auto codecDesc = avcodec_descriptor_get(formatCtx->streams[i]->codecpar->codec_id);
        if (codecDesc) {
          Logger::error(AVtag) << "Muxer[" << index << "]->stream[" << i << "]'s codec is \"" << codecDesc->name
                               << "\" (" << codecDesc->long_name << ")" << std::endl;
        }
      }
    }
    setThreadStatus(MuxerThreadStatus::WriteError);
  }
}

void Muxer::writeFrame(AVPacket* const pkt) {
  if (!formatCtx->pb) {
    Logger::warning(AVtag) << "Can't write video frame because of missing IO context in Format context." << std::endl;
    // set the write error status
    setThreadStatus(MuxerThreadStatus::WriteError);
    return;
  }

  packets.rescaleTimestamp(encoderContexts[pkt->stream_index], formatCtx->streams[pkt->stream_index], pkt);

  // write the compressed frame to the container output file
  int r = av_interleaved_write_frame(formatCtx, pkt);
  if (r < 0) {
    Logger::warning(AVtag) << "Can't write video frame. Error : " << r << " : " << Util::errorString(r) << std::endl;
    // check for socket timeout error, to set the thread status
    if (r == FFMPEG_SOCKET_TIMEOUT_ERROR) {
      setThreadStatus(MuxerThreadStatus::NetworkError);
    } else if (r == -EIO) {
      setThreadStatus(MuxerThreadStatus::WriteError);
    }
  }
}

void Muxer::writeTrailer() {
  int r = av_write_trailer(formatCtx);
  if (r < 0) {
    Logger::error(AVtag) << "Could not write trailer. Error : " << r << " : " << Util::errorString(r) << std::endl;
    setThreadStatus(MuxerThreadStatus::WriteError);
  }
  if (formatCtx && !(formatCtx->flags & AVFMT_NOFILE)) {
    r = avio_closep(&formatCtx->pb);
    if (r < 0) {
      Logger::error(AVtag) << "Could not close muxer. Error : " << r << " : " << Util::errorString(r) << std::endl;
      setThreadStatus(MuxerThreadStatus::WriteError);
    }
  }
}

int64_t Muxer::getMuxedSize() const { return avio_size(formatCtx->pb); }
}  // namespace Output
}  // namespace VideoStitch
