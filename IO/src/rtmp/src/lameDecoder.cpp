// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "lameDecoder.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Input {

MP3Decoder::MP3Decoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth,
                       const uint8_t nbChannels)
    : AudioDecoder(asToFill, samplingRate, samplingDepth, nbChannels), hipLame(hip_decode_init()) {
  if (!hipLame) {
    Logger::get(Logger::Error) << "MP3 DECODER : Unable to open mp3 decoder" << std::endl;
    assert(false);
  }
}

MP3Decoder::~MP3Decoder() { hip_decode_exit(hipLame); }

void MP3Decoder::demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt) {
  avpkt.data = Span<unsigned char>(pkt.data() + 1, pkt.size() - 1);
  avpkt.pts = avpkt.dts = pkt.timestamp * 1000;
}

bool MP3Decoder::decode(IO::DataPacket* pkt) {
  mp3data_struct metaData;

  int64_t nbOut;

  // mp3 to wav compression ratio shouldn't be over 15
  if (pcm_l.size() < pkt->size() * 15 / 2) {
    pcm_l.resize(pkt->size() * 15 / 2);
    pcm_r.resize(pkt->size() * 15 / 2);
  }

  nbOut = hip_decode_headers(hipLame, pkt->data() + 1, pkt->size() - 1, pcm_l.data(), pcm_r.data(), &metaData);
  if (nbOut < 0) {
    Logger::get(Logger::Error) << "MP3 DECODER : decoding error " << nbOut << std::endl;
    return false;
  }

  std::lock_guard<std::mutex> lock(audioStream->audioBufferMutex);

  if (metaData.stereo) {
    uint8_t* ptr_l = reinterpret_cast<uint8_t*>(pcm_l.data());
    uint8_t* ptr_r = reinterpret_cast<uint8_t*>(pcm_r.data());
    for (int64_t i = 0; i < nbOut; i++) {
      audioStream->stream.push(ptr_l, sizeof(int16_t));
      audioStream->stream.push(ptr_r, sizeof(int16_t));
      ptr_l += sizeof(int16_t);
      ptr_r += sizeof(int16_t);
    }
  } else {
    uint8_t* ptr_l = reinterpret_cast<uint8_t*>(pcm_l.data());
    audioStream->stream.push(ptr_l, pcm_l.size() * sizeof(int16_t));
  }

  return true;
}

}  // namespace Input
}  // namespace VideoStitch
