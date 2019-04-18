// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "netStreamReader.hpp"
#include "util.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <stdlib.h>
#include <assert.h>
#include <memory>

static std::string NSRtag("NetStreamReader");

namespace VideoStitch {
namespace Input {

bool netStreamReader::handles(const std::string& filename) { return VideoStitch::Util::isStream(filename); }

netStreamReader::netStreamReader(readerid_t id, const std::string& displayName, const int64_t width,
                                 const int64_t height, const int firstFrame, const AVPixelFormat fmt,
                                 AddressSpace addrSpace, struct AVFormatContext* formatCtx,
#ifdef QUICKSYNC
                                 class QSVContext* qsvCtx,
#endif
                                 struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
                                 struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* videoFrame,
                                 struct AVFrame* audioFrame, Util::TimeoutHandler* interruptCallback,
                                 const signed videoIdx, const signed audioIdx, const Audio::ChannelLayout layout,
                                 const Audio::SamplingRate samplingRate, const Audio::SamplingDepth samplingDepth)
    : Reader(id),
      LibavReader(displayName, width, height, firstFrame, fmt, addrSpace, formatCtx,
#ifdef QUICKSYNC
                  qsvCtx,
#endif
                  videoDecoderCtx, audioDecoderCtx, videoCodec, audioCodec, videoFrame, audioFrame, interruptCallback,
                  videoIdx, audioIdx, layout, samplingRate, samplingDepth),
      stoppingQueues(false),
      frameAvailable(false),
      stoppingFrames(false) {
  frame.resize((size_t)getFrameDataSize());
  memset(frame.data(), 0, frame.size());

  handlePackets = std::thread{&netStreamReader::readNetPackets, this};
  handleVideo = std::thread{&netStreamReader::decodeVideo, this};
  if (audioIdx != INVALID_STREAM_ID) {
    handleAudio = std::thread{&netStreamReader::decodeAudio, this};
  }
}

netStreamReader::~netStreamReader() {
  {
    std::lock_guard<std::mutex> lockV(videoQueueMutex);
    std::lock_guard<std::mutex> lockA(audioQueueMutex);
    stoppingQueues = true;
  }
  cvDecodeVideo.notify_one();
  cvDecodeAudio.notify_one();

  {
    std::lock_guard<std::mutex> lock(videoFrameMutex);
    stoppingFrames = true;
  }
  cvFrameConsumed.notify_one();
  cvNewFrame.notify_one();

  handlePackets.join();
  handleVideo.join();
  if (audioIdx != INVALID_STREAM_ID) {
    handleAudio.join();
  }

  // clear videoPacketQueue
  while (!videoPacketQueue.empty()) {
    auto pkt = videoPacketQueue.front();
    videoPacketQueue.pop();
    av_free_packet(pkt);
    delete pkt;
  }
  // clear audioPacketQueue
  while (!audioPacketQueue.empty()) {
    auto pkt = audioPacketQueue.front();
    audioPacketQueue.pop();
    av_free_packet(pkt);
    delete pkt;
  }
}

// -------------------------- Read packet from network thread --------------------------

void netStreamReader::readNetPackets() {
  while (!stoppingQueues) {
    AVPacket* pkt = new AVPacket();
    av_init_packet(pkt);
    LibavReader::LibavReadStatus packetStatus = readPacket(pkt);
    // always push packet, so that we can clean it up later
    if (pkt->stream_index == videoIdx) {
      std::lock_guard<std::mutex> lock(videoQueueMutex);
      videoPacketQueue.push(pkt);
    } else if (audioIdx != INVALID_STREAM_ID && pkt->stream_index == audioIdx) {
      std::lock_guard<std::mutex> lock(audioQueueMutex);
      audioPacketQueue.push(pkt);
    } else {
      av_packet_unref(pkt);
      delete pkt;
    }

    if (packetStatus == LibavReader::LibavReadStatus::EndOfPackets) {
      Logger::warning(NSRtag) << "End-of-stream reached" << std::endl;
      // stop the decoder
      {
        std::lock_guard<std::mutex> lockV(videoQueueMutex);
        std::lock_guard<std::mutex> lockA(audioQueueMutex);
        stoppingQueues = true;
      }
    }
    cvDecodeVideo.notify_one();
    cvDecodeAudio.notify_one();
  }
}

// -------------------------- Decoding thread --------------------------

void netStreamReader::decodeVideo() {
  while (!stoppingQueues) {
    // wait for last frame to be consumed
    bool gotPicture = false;
    {
      std::unique_lock<std::mutex> lock(videoFrameMutex);
      if (frameAvailable) {
        cvFrameConsumed.wait(lock, [this]() { return !frameAvailable || stoppingFrames; });
        if (stoppingFrames) {
          break;
        }
      }
      // first check if the video decoder has ready frames
      decodeVideoPacket(&gotPicture, nullptr, frame.data());
      if (gotPicture) {
        frameAvailable = true;
        lock.unlock();
        cvNewFrame.notify_one();
        continue;
      }
    }
    // if no frame is available, send a new packet to the decoder
    AVPacket* pkt = nullptr;
    {
      std::unique_lock<std::mutex> lock(videoQueueMutex);
      cvDecodeVideo.wait(lock, [this] { return !videoPacketQueue.empty() || stoppingQueues; });
      if (stoppingQueues) {
        break;
      }
      pkt = videoPacketQueue.front();
      videoPacketQueue.pop();
    }

    {
      std::unique_lock<std::mutex> lock(videoFrameMutex);
      decodeVideoPacket(&gotPicture, pkt, frame.data());
      if (gotPicture) {
        frameAvailable = true;
        lock.unlock();
        cvNewFrame.notify_one();
      }
    }
    av_packet_unref(pkt);
    delete pkt;
  }
}

void netStreamReader::decodeAudio() {
  while (!stoppingQueues) {
    AVPacket* pkt = nullptr;
    {
      std::unique_lock<std::mutex> lock(audioQueueMutex);
      cvDecodeAudio.wait(lock, [this] { return !audioPacketQueue.empty() || stoppingQueues; });
      if (stoppingQueues) {
        break;
      }

      pkt = audioPacketQueue.front();
      audioPacketQueue.pop();
    }
    {
      std::lock_guard<std::mutex> lock(audioBufferMutex);
      decodeAudioPacket(pkt);
    }
    av_packet_unref(pkt);
    delete pkt;
  }
}

// -------------------------- Stitcher thread --------------------------

ReadStatus netStreamReader::readFrame(mtime_t& date, unsigned char* video) {
  VideoStitch::Util::SimpleProfiler prof("FFmpeg in : read frame ", true, Logger::get(Logger::Debug));
  {
    std::unique_lock<std::mutex> lock(videoFrameMutex);
    // from std::condition_variable::wait_for
    // return false if the predicate pred still evaluates to false after the rel_time timeout expired
    if (!cvNewFrame.wait_for(lock, std::chrono::milliseconds(1000),
                             [this] { return frameAvailable || stoppingFrames; })) {
      return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
    }
    // cvNewFrame was awakened, chek if we reached end of streaming
    if (stoppingFrames) {
      return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
    }
    // Video reading
    memcpy(video, frame.data(), frame.size());
    date = videoTimeStamp;
    frameAvailable = false;
  }
  cvFrameConsumed.notify_one();

  return ReadStatus::OK();
}

ReadStatus netStreamReader::readSamples(size_t nbSamples, Audio::Samples& audioSamples) {
  std::lock_guard<std::mutex> lock(audioBufferMutex);
  return LibavReader::readSamples(nbSamples, audioSamples);
}

}  // namespace Input
}  // namespace VideoStitch
