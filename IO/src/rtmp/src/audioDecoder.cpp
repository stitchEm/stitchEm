// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioDecoder.hpp"
#include "aacDecoder.hpp"
#include "lameDecoder.hpp"
#include "pcmDecoder.hpp"
#include "libvideostitch/logging.hpp"

static const AVal av_aac = mAVC("m4a");

namespace VideoStitch {
namespace Input {

AudioDecoder::AudioDecoder(AudioStream* asToFill, const uint64_t samplingRate, const uint8_t samplingDepth,
                           const uint8_t nbChannels)
    : samplingRate(samplingRate), samplingDepth(samplingDepth), numberOfChannels(nbChannels), audioStream(asToFill) {}

std::unique_ptr<AudioDecoder> AudioDecoder::createAudioDecoder(AMFDataType encoderType, AMFObjectProperty* amfOProperty,
                                                               AudioStream* audioStream, const long samplingRate,
                                                               const int samplingDepth, const int nbChannels) {
  Logger::get(Logger::Info) << "RTMP Inputs : samplingRate " << samplingRate << std::endl;
  Logger::get(Logger::Info) << "RTMP Inputs : sample depth " << samplingDepth << std::endl;
  Logger::get(Logger::Info) << "RTMP Inputs : number of channels " << nbChannels << std::endl;

  if (encoderType == AMF_NUMBER) {
    int codecId = int(AMFProp_GetNumber(amfOProperty));
    Logger::get(Logger::Info) << "RTMP Inputs : audiocodecID " << codecId << std::endl;

    switch (codecId) {
      case 2: {
        return std::make_unique<MP3Decoder>(audioStream, samplingRate, samplingDepth, nbChannels);
      }
      case 3: {
        return std::make_unique<PCMDecoder>(audioStream, samplingRate, samplingDepth, nbChannels);
      }
      case 10: {
        return std::make_unique<AACDecoder>(audioStream, samplingRate, samplingDepth, nbChannels);
      }
      default:
        break;
    }
  } else {
    AVal stringValue;
    AMFProp_GetString(amfOProperty, &stringValue);
    Logger::get(Logger::Info) << "RTMP Inputs : audiocodecID " << stringValue.av_val << std::endl;

    if (AVMATCH(&stringValue, &av_aac)) {
      return std::make_unique<AACDecoder>(audioStream, samplingRate, samplingDepth, nbChannels);
    }
  }

  return nullptr;
}

}  // namespace Input
}  // namespace VideoStitch
