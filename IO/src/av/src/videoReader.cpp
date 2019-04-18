// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoReader.hpp"
#include "util.hpp"
#include "extensionChecker.hpp"
#include "io.hpp"
#include "libvideostitch/logging.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace VideoStitch {
namespace Input {

static std::string AVtag("videoinput");

bool FFmpegReader::handles(const std::string& filename) {
  // Ignore images or streams
  if (hasExtension(filename, ".png") || hasExtension(filename, ".jpg") || hasExtension(filename, ".jpeg") ||
#ifdef __ANDROID__
      hasExtension(filename, ".mp4") || /* use mp4Plugin for HW acceleration */
#endif
      hasExtension(filename, ".wav") || VideoStitch::Util::isStream(filename)) {
    return false;
  }

  AVProbeData pd = {};
  const size_t size = 1024;
  pd.buf = new unsigned char[size + AVPROBE_PADDING_SIZE];
  pd.filename = nullptr;

  FILE* f = VideoStitch::Io::openFile(filename, "rb");
  if (!f) {
    Logger::error(AVtag) << "Couldn't open file " << filename << " for probing. Aborting." << std::endl;
    delete[] pd.buf;
    return false;
  }
  const size_t bytesRead = fread(pd.buf, 1, size, f);
  pd.buf_size = (int)bytesRead;
  fclose(f);
  if (bytesRead < size) {
    Logger::warning(AVtag) << "Could only read " << bytesRead << "bytes (instead of " << size << ") for probing format."
                           << std::endl;
  }
  memset(pd.buf + bytesRead, 0, AVPROBE_PADDING_SIZE);

  // can libav guess format?
  Util::Libav::checkInitialization();
  Util::Libav::Lock sl;
  AVInputFormat* format = av_probe_input_format(&pd, 1);
  delete[] pd.buf;

  return format != nullptr;
}

FFmpegReader::FFmpegReader(readerid_t id, const std::string& displayName, const int64_t width, const int64_t height,
                           const int firstFrame, const AVPixelFormat fmt, AddressSpace addrSpace,
                           struct AVFormatContext* formatCtx,
#ifdef QUICKSYNC
                           class QSVContext* qsvCtx,
#endif
                           struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
                           struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* videoFrame,
                           struct AVFrame* audioFrame, Util::TimeoutHandler* interruptCallback, const int videoIdx,
                           const int audioIdx, const Audio::ChannelLayout layout,
                           const Audio::SamplingRate samplingRate, const Audio::SamplingDepth samplingDepth)
    : Reader(id),
      LibavReader(displayName, width, height, firstFrame, fmt, addrSpace, formatCtx,
#ifdef QUICKSYNC
                  qsvCtx,
#endif
                  videoDecoderCtx, audioDecoderCtx, videoCodec, audioCodec, videoFrame, audioFrame, interruptCallback,
                  videoIdx, audioIdx, layout, samplingRate, samplingDepth) {
  frame.resize((size_t)getFrameDataSize());
  memset(frame.data(), 0, frame.size());
}

FFmpegReader::~FFmpegReader() {
  for (auto pkt : videoQueue) {
    av_packet_unref(&pkt);
  }
  for (auto pkt : audioQueue) {
    av_packet_unref(&pkt);
  }
}

Status FFmpegReader::seekFrame(frameid_t targetFrame) {
  std::unique_lock<std::recursive_mutex> lk(monitor);

  // first clear the prerolled data
  // first empty the queues, then clear the decoded audioBuffer
  AVPacket pkt;
  while (videoQueue.size() > 0) {
    pkt = videoQueue.front();
    videoQueue.pop_front();
    av_packet_unref(&pkt);
  }
  while (audioQueue.size() > 0) {
    pkt = audioQueue.front();
    audioQueue.pop_front();
    av_packet_unref(&pkt);
  }
  for (auto& q : audioBuffer) {
    q.clear();
  }
  nbSamplesInAudioBuffer = 0;
  audioTimeStamp = -1;

  // then seek to the correct frame
  Status st = LibavReader::seekFrame(targetFrame);

  // finally drop audio until aligned with the video
  while (audioQueue.size() > 0 && av_compare_ts(audioQueue.front().pts, formatCtx->streams[audioIdx]->time_base,
                                                currentVideoPts, formatCtx->streams[videoIdx]->time_base) < 0) {
    pkt = audioQueue.front();
    audioQueue.pop_front();
    av_packet_unref(&pkt);
  }

  return st;
}

ReadStatus FFmpegReader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  std::unique_lock<std::recursive_mutex> lk(monitor);

  AVPacket pkt;
  bool got_picture = false;

  while (!got_picture) {
    // first try to read available frames from the decoder
    decodeVideoPacket(&got_picture, nullptr, videoFrame);
    // if no decoded frame is available, try pushing a new packet to the decoder
    if (!got_picture) {
      bool eof = false;
      if (videoQueue.size() == 0) {
        // we're gonna read a packet from the video queue, fill both queues
        // until we're putting another video packet in the queue or
        // reaching EOF
        while ((!eof) && (videoQueue.empty())) {
          switch (readPacket(&pkt)) {
            case LibavReader::LibavReadStatus::Error:
              // TODOLATERSTATUS propagate error cause
              return Status{Origin::Input, ErrType::RuntimeError, "FFmpegReader failed to read frame"};
            case LibavReader::LibavReadStatus::EndOfPackets:
              eof = true;  // exit loop, we emptied the container
              break;
            case LibavReader::LibavReadStatus::Ok:
              if (pkt.stream_index == videoIdx) {
                videoQueue.push_back(pkt);
              } else if (pkt.stream_index == audioIdx) {
                audioQueue.push_back(pkt);
              } else {
                Logger::warning(AVtag) << "Unknown av stream with index " << pkt.stream_index << ", ignoring."
                                       << std::endl;
              }
              break;
            default:
              assert(false);
              Logger::error(AVtag) << "FFmpegReader::readFrame(): Implementation Error" << std::endl;
              break;
          }
        }
      }

      if (videoQueue.size() == 0) {  // equivalent to eof = true
        assert(eof == true);
        // flush
        flushVideoDecoder(&got_picture, videoFrame);
        if (!got_picture) {
          // actual end-of-file
          return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
        }
      } else {
        // feed a video packet to the decoder, repeat all until getting a picture
        pkt = videoQueue.front();
        videoQueue.pop_front();
        decodeVideoPacket(&got_picture, &pkt, videoFrame);
        av_packet_unref(&pkt);
      }
    }
  }

  date = videoTimeStamp;

  Logger::debug(AVtag) << "Reader " << id << " read a frame at " << videoTimeStamp
                       << ", video queue size: " << videoQueue.size() << std::endl;

  return ReadStatus::OK();
}

size_t FFmpegReader::available() {
  std::unique_lock<std::recursive_mutex> lk(monitor);
  // 4096 samples should be more than enough to fill an audio frame
  ensureAudio(4096);
  return LibavReader::available();
}

bool FFmpegReader::ensureAudio(size_t nbSamples) {
  // the goal of this function is to ensure nbSamples can be read from the
  // reader.
  // return false if reached EOF before ensuring enough samples.
  AVPacket pkt;
  bool eof = false;

  while (nbSamplesInAudioBuffer < nbSamples && !eof) {
    if (audioQueue.size() == 0) {
      // we're gonna read a packet from the audio queue, fill both queues
      // until we're putting another audio packet in the queue or
      // reaching EOF
      while ((!eof) && (audioQueue.empty())) {
        switch (readPacket(&pkt)) {
          case LibavReader::LibavReadStatus::Error:
            // TODOLATERSTATUS propagate error cause
            return false;
          case LibavReader::LibavReadStatus::EndOfPackets:
            eof = true;  // exit loop, we emptied the container
            break;
          case LibavReader::LibavReadStatus::Ok:
            if (pkt.stream_index == audioIdx) {
              audioQueue.push_back(pkt);
            } else if (pkt.stream_index == videoIdx) {
              videoQueue.push_back(pkt);
            } else {
              Logger::warning(AVtag) << "Unknown av stream with index " << pkt.stream_index << ", ignoring."
                                     << std::endl;
            }
            break;
        }
      }
    }

    if (audioQueue.size() == 0) {  // equivalent to eof = true
      assert(eof == true);
      // flush
      decodeAudioPacket(nullptr, true);
    } else {
      // feed an audio packet to the decoder, repeat all until getting enough samples
      pkt = audioQueue.front();
      audioQueue.pop_front();
      decodeAudioPacket(&pkt);
      av_packet_unref(&pkt);
    }
  }

  return eof;
}

ReadStatus FFmpegReader::readSamples(size_t nbSamples, Audio::Samples& audioSamples) {
  std::unique_lock<std::recursive_mutex> lk(monitor);
  bool eof = ensureAudio(nbSamples);
  if (eof && nbSamplesInAudioBuffer == 0) {
    audioSamples = Audio::Samples();
    return ReadStatus::OK();
  }
  Logger::debug(AVtag) << "Reader " << id << " read " << nbSamples << " samples starting at " << audioTimeStamp
                       << ", audio queue size: " << audioQueue.size() << std::endl;

  // call the legacy readSamples function
  return LibavReader::readSamples(nbSamples, audioSamples);
}

}  // namespace Input
}  // namespace VideoStitch
