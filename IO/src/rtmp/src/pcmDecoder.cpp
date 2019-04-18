// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pcmDecoder.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Input {

static const size_t kMaxBufferSize{10000000};  // ~1 minute of 44.1kHz/16-bit stereo

PCMDecoder::PCMDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth,
                       const uint8_t nbChannels)
    : AudioDecoder(asToFill, samplingRate, samplingDepth, nbChannels), warn(true) {}

bool PCMDecoder::decode(IO::DataPacket* pkt) {
  Logger::get(Logger::Debug) << "PCM decoder: decoding packet with timestamp: " << pkt->timestamp << std::endl;

  std::lock_guard<std::mutex> lock(audioStream->audioBufferMutex);

  // Calculate new timestamp
  size_t samplesPerChannel = audioStream->stream.size() / (size_t)(numberOfChannels * samplingDepth / 8);
  // Convert from ms to us
  mtime_t mts = ((mtime_t)pkt->timestamp * 1000);
  // Subtract time of this (current) packet
  mts -= (mtime_t)(samplesPerChannel * 1000000UL / samplingRate);

  // Check for lost packets (too big a jump in timestamp)
  if ((audioStream->cnts != 0) && (std::abs(mts - CNTOMTIME(audioStream->cnts)) > 1000)) {
    Logger::get(Logger::Warning) << "PCM decoder : audio packet lost, resync : "
                                 << (mts - CNTOMTIME(audioStream->cnts)) / 1000 << " ms " << std::endl;
    // Realign timestamp according to arrived packet
    audioStream->cnts = MTOCNTIME(mts);
  }

  // Don't allow audioStream to grow unbounded.
  if (audioStream->stream.size() + pkt->size() - 1 > kMaxBufferSize) {
    if (warn) {
      warn = false;  // Avoid flooding logger
      Logger::get(Logger::Warning) << "[RTMP::PCMDecoder::decode] Dropping audio to fit new packet" << std::endl;
    }
    audioStream->stream.erase(pkt->size() - 1);
  } else {
    warn = true;
  }
  // -1 due to a non-audio byte in packet.
  audioStream->stream.push(pkt->data() + 1, pkt->size() - 1);

  return true;
}

void PCMDecoder::demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt) {
  avpkt.data = Span<unsigned char>(pkt.data() + 1, pkt.size() - 1);
  avpkt.pts = avpkt.dts = pkt.timestamp * 1000;
}

std::string PCMDecoder::name() {
  if (samplingDepth == 8) {
    return "pcm_u8";
  } else {
    return "pcm_s16le";
  }
}

}  // namespace Input
}  // namespace VideoStitch
