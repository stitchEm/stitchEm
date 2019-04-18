// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "aacDecoder.hpp"
#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Input {

AACDecoder::AACDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth,
                       const uint8_t nbChannels)
    : AudioDecoder(asToFill, samplingRate, samplingDepth, nbChannels), faad(NeAACDecOpen()), initialised(false) {
  NeAACDecConfigurationPtr conf = NeAACDecGetCurrentConfiguration(faad);
  // Logger::get(Logger::Info) << "AAC Decoder parameter " <<  (int)conf->defObjectType << " samplerate : " <<
  // conf->defSampleRate << " sampledepth : " << (int)conf->outputFormat << std::endl;

  conf->defSampleRate = (unsigned long)samplingRate;

  char err = NeAACDecSetConfiguration(faad, conf);
  if (err == 0) {
    // Handle error
    Logger::get(Logger::Error) << "AAC decoder : NeAACDecSetConfiguration error: " << (int)err << std::endl;
  }

  conf = NeAACDecGetCurrentConfiguration(faad);
  Logger::get(Logger::Info) << "AAC Decoder parameter "
                            << " samplerate : " << conf->defSampleRate << " sampledepth : " << (int)conf->outputFormat
                            << std::endl;
}

AACDecoder::~AACDecoder() { NeAACDecClose(faad); }

void AACDecoder::demux(VideoStitch::IO::DataPacket& pkt, VideoStitch::IO::Packet& avpkt) {
  avpkt.data = Span<unsigned char>(pkt.data() + 2, pkt.size() - 2);
  avpkt.pts = avpkt.dts = pkt.timestamp * 1000;
}

bool AACDecoder::decode(IO::DataPacket* pkt) {
  NeAACDecFrameInfo hInfo = {};

  /*Init decoder with in-stream parameter*/
  if ((!initialised) && (pkt->data()[1] == 0)) {
    unsigned long longSampleRate = (unsigned long)samplingRate;
    // Skip 2 byte AAC header when passing data to decoder
    auto err = NeAACDecInit(faad, pkt->data() + 2, (unsigned long)pkt->size() - 2, &longSampleRate, &numberOfChannels);
    if (err < 0) {
      // Handle error
      Logger::get(Logger::Error) << "AAC decoder : NeAACDecInit error: " << (int)err << std::endl;
    }
    initialised = true;
    return false;
  }

  /*Decode*/
  if ((initialised) && (pkt->data()[1] == 1)) {
    unsigned long nbIn = 0;
    do {
      void* out = NeAACDecDecode(faad, &hInfo, pkt->data() + nbIn + 2, (unsigned long)pkt->size() - nbIn - 2);
      if (out == nullptr) {
        Logger::get(Logger::Error) << "AAC decoder : NeAACDecDecode error: " << (int)hInfo.error << std::endl;
      }
      nbIn += hInfo.bytesconsumed;

      std::lock_guard<std::mutex> lk(audioStream->audioBufferMutex);
      audioStream->stream.push((uint8_t*)out, hInfo.samples * samplingDepth / 8);

    } while (nbIn < pkt->size() - 2);

    return true;
  } else {
    return false;
  }
}

}  // namespace Input
}  // namespace VideoStitch
