// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "lameEncoder.hpp"
#include "amfIncludes.hpp"

#include <cmath>

namespace VideoStitch {
namespace Output {

const int MP3Encoder::DEFAULT_AUDIO_BITRATE = 64;
const int MP3Encoder::audioBlockSize = 8;  // 2 channels * 32bit samples
const int MP3Encoder::frameSize = 1152;

const AVal MP3Encoder::av_audiocodecid = mAVC("audiocodecid");
const AVal MP3Encoder::av_audiodatarate = mAVC("audiodatarate");
const AVal MP3Encoder::av_audiosamplerate = mAVC("audiosamplerate");
const AVal MP3Encoder::av_audiosamplesize = mAVC("audiosamplesize");
const AVal MP3Encoder::av_audiochannels = mAVC("audiochannels");
const AVal MP3Encoder::av_stereo = mAVC("stereo");

MP3Encoder::MP3Encoder(unsigned int bitRate, Audio::SamplingRate sampleRate, int nbChans, Audio::SamplingDepth fmt)
    : bitRate(bitRate), sampleRate(getIntFromSamplingRate(sampleRate)), nbChans(nbChans), fmt(fmt) {
  lgf = lame_init();
  if (!lgf) {
    Logger::get(Logger::Error) << "RTMP : Unable to open mp3 encoder" << std::endl;
    assert(false);
  }

  lame_set_in_samplerate(lgf, Audio::getIntFromSamplingRate(sampleRate));
  lame_set_out_samplerate(lgf, Audio::getIntFromSamplingRate(sampleRate));
  lame_set_num_channels(lgf, nbChans);
  lame_set_disable_reservoir(lgf, TRUE);  // bit reservoir has to be disabled for seamless streaming
  lame_set_quality(lgf, 2);
  lame_set_VBR(lgf, vbr_off);
  lame_set_brate(lgf, bitRate);
  lame_init_params(lgf);

  // FLV AudioTagHeader
  switch (sampleRate) {
    case Audio::SamplingRate::SR_22050:
      mp3buf.push_back(0x2b);  // codec id tag, 16bits stereo at 22050Hz
      break;
    case Audio::SamplingRate::SR_44100:
      mp3buf.push_back(0x2f);  // codec id tag, 16bits stereo at 44100Hz
      break;
    case Audio::SamplingRate::SR_NONE:
    case Audio::SamplingRate::SR_32000:
    case Audio::SamplingRate::SR_48000:
    case Audio::SamplingRate::SR_88200:
    case Audio::SamplingRate::SR_96000:
    case Audio::SamplingRate::SR_176400:
    case Audio::SamplingRate::SR_192000:
      assert(false);
      break;
  }
}

char* MP3Encoder::metadata(char* enc, char* pend) {
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiocodecid, 2.);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiodatarate, double(bitRate));  // ex. 128kb\s
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiosamplerate, (double)sampleRate / 1000);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiosamplesize, 16.0);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_audiochannels, double(nbChans));
  enc = AMF_EncodeNamedBoolean(enc, pend, &av_stereo, nbChans == 2);
  return enc;
}

bool MP3Encoder::encode(mtime_t date, uint8_t* const* input, unsigned int numInputFrames,
                        std::vector<VideoStitch::IO::DataPacket>& packets) {
  uint8_t* leftptr = input[0];
  uint8_t* rightptr = input[1];

  mtime_t currentDate = date;

  int mp3buf_size = uint32_t(1.25 * (frameSize + 7200.0));
  mp3buf.resize(mp3buf_size + 1);

  int q = numInputFrames / frameSize;
  int r = numInputFrames % frameSize;
  for (int i = 0; i <= q; ++i) {
    int nbSamples = (i == q ? r : frameSize);

    int buf_size = 0;
    switch (fmt) {
      case Audio::SamplingDepth::INT16:
        buf_size = lame_encode_buffer_interleaved(lgf, (int16_t*)leftptr, nbSamples, mp3buf.data() + 1, mp3buf_size);
        leftptr += sizeof(int16_t) * nbSamples * nbChans;
        break;
      case Audio::SamplingDepth::FLT:
        buf_size =
            lame_encode_buffer_interleaved_ieee_float(lgf, (float*)leftptr, nbSamples, mp3buf.data() + 1, mp3buf_size);
        leftptr += sizeof(float) * nbSamples * nbChans;
        break;
      case Audio::SamplingDepth::DBL:
        buf_size = lame_encode_buffer_interleaved_ieee_double(lgf, (double*)leftptr, nbSamples, mp3buf.data() + 1,
                                                              mp3buf_size);
        leftptr += sizeof(double) * nbSamples * nbChans;
        break;
      case Audio::SamplingDepth::INT16_P:
        buf_size =
            lame_encode_buffer(lgf, (int16_t*)leftptr, (int16_t*)rightptr, nbSamples, mp3buf.data() + 1, mp3buf_size);
        leftptr += sizeof(int16_t) * nbSamples;
        rightptr += sizeof(int16_t) * nbSamples;
        break;
      case Audio::SamplingDepth::INT32_P:
        buf_size = lame_encode_buffer_int(lgf, (int32_t*)leftptr, (int32_t*)rightptr, nbSamples, mp3buf.data() + 1,
                                          mp3buf_size);
        leftptr += sizeof(int32_t) * nbSamples;
        rightptr += sizeof(int32_t) * nbSamples;
        break;
      case Audio::SamplingDepth::FLT_P:
        buf_size = lame_encode_buffer_ieee_float(lgf, (float*)leftptr, (float*)rightptr, nbSamples, mp3buf.data() + 1,
                                                 mp3buf_size);
        leftptr += sizeof(float) * nbSamples;
        rightptr += sizeof(float) * nbSamples;
        break;
      case Audio::SamplingDepth::DBL_P:
        buf_size = lame_encode_buffer_ieee_double(lgf, (double*)leftptr, (double*)rightptr, nbSamples,
                                                  mp3buf.data() + 1, mp3buf_size);
        leftptr += sizeof(double) * nbSamples;
        rightptr += sizeof(double) * nbSamples;
        break;
      case Audio::SamplingDepth::INT32:
      case Audio::SamplingDepth::UINT8:
      case Audio::SamplingDepth::UINT8_P:
      default:
        return false;
    }

    if (buf_size <= 0) {
      if (buf_size < 0) Logger::get(Logger::Error) << "RTMP : MP3 encode failed" << std::endl;
      return false;
    }

    IO::DataPacket packet(mp3buf.data(), buf_size + 1);
    packet.timestamp = mtime_t(std::round(currentDate / 1000.0));
    packet.type = IO::PacketType_Audio;
    packets.push_back(packet);

    currentDate += (nbSamples * 1000000) / sampleRate;
  }

  return true;
}

std::unique_ptr<AudioEncoder> MP3Encoder::createMP3Encoder(const Ptv::Value& config,
                                                           const Audio::SamplingRate samplingRate,
                                                           Audio::SamplingDepth depth,
                                                           const Audio::ChannelLayout layout) {
  INT(config, audio_bitrate, MP3Encoder::DEFAULT_AUDIO_BITRATE);
  int nb_channels = Audio::getNbChannelsFromChannelLayout(layout);
  return std::unique_ptr<AudioEncoder>(new MP3Encoder(audio_bitrate, samplingRate, nb_channels, depth));
}

}  // namespace Output
}  // namespace VideoStitch
