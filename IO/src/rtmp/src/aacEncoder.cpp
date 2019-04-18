// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <stdlib.h>
#include "aacEncoder.hpp"
#include "amfIncludes.hpp"

namespace VideoStitch {
namespace Output {

const int AACEncoder::DEFAULT_AUDIO_BITRATE = 64;

const AVal AACEncoder::av_audiocodecid = mAVC("audiocodecid");
const AVal AACEncoder::av_mp4a = mAVC("mp4a");
const AVal AACEncoder::av_audiodatarate = mAVC("audiodatarate");
const AVal AACEncoder::av_audiosamplerate = mAVC("audiosamplerate");
const AVal AACEncoder::av_audiosamplesize = mAVC("audiosamplesize");
const AVal AACEncoder::av_audiochannels = mAVC("audiochannels");
const AVal AACEncoder::av_stereo = mAVC("stereo");
const std::string AacEncTag = "AACEncoder";
static const size_t MAX_CHANNEL_MAP = 64;  // Maximum channel map size (see faaccfg.h)
AACEncoder::AACEncoder(unsigned int bitRate, int sampleRate, int nbChans, Audio::SamplingDepth fmt,
                       const std::vector<int64_t>& channelMap)
    : bitRate(bitRate), sampleRate(sampleRate), nbChans(nbChans), sampleFormat(fmt) {
  sampleDepth = (int)getSampleSizeFromSamplingDepth(fmt);

  unsigned long maxOutputBytes;

  faac = faacEncOpen(sampleRate, nbChans, &inputSamples, &maxOutputBytes);

  faacEncConfigurationPtr config = faacEncGetCurrentConfiguration(faac);
  config->bitRate = (bitRate * 1000) / nbChans;
  config->quantqual = 100;                 // Variable bit rate (VBR) quantizer quality in %
  config->inputFormat = FAAC_INPUT_FLOAT;  // All formats will be converted to float
  config->mpegVersion = MPEG4;
  config->aacObjectType = LOW;
  config->useLfe = 0;
  config->outputFormat = 0;
  // As the internal lib supports WXYZ order, we apply this remapping to support the ambix format WYZX
  // We consider that layout with 4 channels is an ambisonic format. The same consideration is done
  // in the avWriter when we add the spatial metadata (see qt-faststart.cpp)
  if (nbChans == 4) {
    config->channel_map[0] = 0;
    config->channel_map[1] = 2;
    config->channel_map[2] = 3;
    config->channel_map[3] = 1;
  }

  // LIBFAAC remaps channels for internal optimization (see faaccfg.h)
  // Here is the channel remapping to apply to compensate this internal optimization.
  if (nbChans == 4 && channelMap.empty()) {
    config->channel_map[0] = 2;
    config->channel_map[1] = 0;
    config->channel_map[2] = 1;
    config->channel_map[3] = 3;
  } else if (channelMap.size() < MAX_CHANNEL_MAP) {
    for (size_t i = 0; i < channelMap.size(); i++) {
      config->channel_map[i] = static_cast<int>(channelMap[i]);
    }
  }

  int ret = faacEncSetConfiguration(faac, config);
  if (!ret) {
    Logger::get(Logger::Error) << "RTMP : Unable to open aac encoder" << std::endl;
    assert(false);
  }
  unsigned char* decoderInfo;
  unsigned long len;
  faacEncGetDecoderSpecificInfo(faac, &decoderInfo, &len);

  // FLV AudioTagHeader
  // https://www.adobe.com/content/dam/Adobe/en/devnet/flv/pdfs/video_file_format_spec_v10.pdf
  aacbuf.push_back(0xaf);  // codec id tag, AAC
  aacbuf.push_back(0x1);   // AAC raw
  aacbuf.resize(maxOutputBytes + 2);

  std::vector<unsigned char> header;
  header.push_back(0xaf);  // codec id tag, AAC
  header.push_back(0x00);  // sequence header
  header.insert(header.end(), decoderInfo, decoderInfo + len);
  headerPkt = VideoStitch::IO::DataPacket(header);
  headerPkt.timestamp = 0;
  headerPkt.type = VideoStitch::IO::PacketType_Audio;

  free(decoderInfo);
}

char* AACEncoder::metadata(char* enc, char* pend) {
  enc = AMF_EncodeNamedString(enc, pend, &av_audiocodecid, &av_mp4a);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiodatarate, double(bitRate));  // ex. 128kb\s
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiosamplerate, sampleRate);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiosamplesize, 16.0);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiochannels, double(nbChans));
  enc = AMF_EncodeNamedBoolean(enc, pend, &av_stereo, nbChans == 2);
  return enc;
}

// FAAC expects values from -32768 to +32767 for
// 16-bit and float, and -8388608 to +8388607 for
// 32-bit audio (a 24 bit value).
//
// We get values from -1.0 to +1.0 for float,
// -32768 to +32767 for 16-bit, and -2^31 to
// +2^31-1 for 32-bit.

namespace {
inline float convertFloatToFloat(uint8_t* sample) {
  float val = *((float*)sample);
  if (val > 0) {
    return float(val * 32767.0);
  } else {
    return float(val * 32768.0);
  }
}
inline float convertInt32ToFloat(uint8_t* sample) {
  int32_t s32 = *((int32_t*)sample);  // Cast to real size and get value
  if (s32 > 0) {
    return (float)((double)s32 * 0.00001525832340831790);  // Convert from 32-bit range to 16-bit range
  } else {
    return (float)((double)s32 * 0.00001525878906250000);
  }
}
}  // namespace

void AACEncoder::stuffInputBuffer(uint8_t* input, int numSamples) {
  switch (sampleFormat) {
    case Audio::SamplingDepth::FLT: {
      for (int i = 0; i < numSamples; ++i) {
        inputBuffer.push_back(convertFloatToFloat(input));
        input += sampleDepth;
      }
      break;
    }
    case Audio::SamplingDepth::INT32: {
      for (int i = 0; i < numSamples; ++i) {
        inputBuffer.push_back(convertInt32ToFloat(input));
        input += sampleDepth;
      }
      break;
    }
    case Audio::SamplingDepth::INT16: {
      for (int i = 0; i < numSamples; ++i) {
        const int16_t s16 = *((int16_t*)input);  // Cast to real size and get value
        inputBuffer.push_back(static_cast<float>(s16));
        input += sampleDepth;
      }
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

/// \fn bool AACEncoder::encode(mtime_t date, uint8_t** input, unsigned int numInputFrames,
/// std::vector<VideoStitch::IO::DataPacket> &packets) \brief Encode a block of audio samples as AAC for RTMP (FLV).
/// \param date The timestamp of the input samples.
/// \param numInputFrames The number of samples per input channel to be encoded.
/// \param packets A vector of output packets to be sent over RTMP.
/// \return `true` on success or `false` if encoding failed.
bool AACEncoder::encode(mtime_t date, uint8_t* const* input, unsigned int numInputFrames,
                        std::vector<VideoStitch::IO::DataPacket>& packets) {
  // The date of the first sample of the encoder's buffer
  // is evaluated as the date of the first sample we are going to insert
  // minus the length of the encoder's buffer
  mtime_t bufferDate = date - (inputBuffer.size() * 1000000) / (nbChans * sampleRate);

  // The number of samples in this function is to be understood
  // as the total number of samples for all channels
  stuffInputBuffer(input[0], numInputFrames * nbChans);

  // Length (in usec) of an encoder frame
  mtime_t frameLength = (inputSamples * 1000000) / (nbChans * sampleRate);

  while (inputBuffer.size() >= inputSamples) {
    int buf_size = faacEncEncode(faac, (int32_t*)inputBuffer.data(), (unsigned int)inputSamples, aacbuf.data() + 2,
                                 (unsigned int)aacbuf.size() - 2);
    framesFed++;
    inputBuffer.erase(inputBuffer.begin(), inputBuffer.begin() + inputSamples);
    bufferDate += frameLength;

    if (buf_size == 0) {
      continue;
    }
    if (buf_size < 0) {
      Logger::get(Logger::Error) << "RTMP : AAC encode failed" << std::endl;
      return false;
    }

    VideoStitch::IO::DataPacket packet(aacbuf.data(), buf_size + 2);
    packet.timestamp = (bufferDate - framesFed * frameLength) / 1000;
    packet.type = VideoStitch::IO::PacketType_Audio;
    packets.push_back(packet);

    framesFed--;
  }

  return true;
}

/// \fn std::unique_ptr<AudioEncoder> AACEncoder::createAACEncoder
/// \brief Open a new AAC encoder object.
/// \param config PTV object with encoder configuration.
/// \param rate Sample rate of incoming signal.
/// \param depth Format of incoming samples.
/// \param layout Channel layout of incoming stream.
/// \return A AACEncoder object on success, or nullptr on failure.
std::unique_ptr<AudioEncoder> AACEncoder::createAACEncoder(const Ptv::Value& config, const Audio::SamplingRate rate,
                                                           const Audio::SamplingDepth depth,
                                                           const Audio::ChannelLayout layout) {
  INT(config, audio_bitrate, AACEncoder::DEFAULT_AUDIO_BITRATE);

  // TODO: Add more supported channel layouts for AAC
  //       (it supports up to 48 or even 64 channels)
  int nb_channels = Audio::getNbChannelsFromChannelLayout(layout);
  if (nb_channels == 0) {
    return nullptr;
  }

  int samplingRate = Audio::getIntFromSamplingRate(rate);
  if (samplingRate > 96000 || samplingRate < 22050) {
    assert(false);
    return nullptr;
  }

  // Channel map
  std::vector<int64_t> channel_map;
  if (Parse::populateIntList("RTMP", config, "channel_map", channel_map, false) == Parse::PopulateResult_WrongType) {
    Logger::error(AacEncTag) << "Audio channel map: : invalid" << std::endl;
    return nullptr;
  }
  if (!channel_map.empty() && int(channel_map.size()) != nb_channels) {
    Logger::error(AacEncTag) << "Audio channel map: : invalid number of elements, does not match channel layout "
                             << Audio::getStringFromChannelLayout(layout) << std::endl;
    return nullptr;
  }

  if (!channel_map.empty()) {
    for (auto& map : channel_map) {
      if (!(0 <= map && map < nb_channels)) {
        Logger::error(AacEncTag) << "Audio channel map: has invalid value " << map << std::endl;
        return nullptr;
      }
    }
  }

  return std::unique_ptr<AudioEncoder>(new AACEncoder(audio_bitrate, samplingRate, nb_channels, depth, channel_map));
}

}  // namespace Output
}  // namespace VideoStitch
