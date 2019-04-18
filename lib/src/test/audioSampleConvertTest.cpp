// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "common/audioUtils.hpp"

#include "libvideostitch/audio.hpp"
#include <audio/converter.hpp>
#include <audio/resampler.hpp>

#include <cmath>
#include <iostream>

namespace VideoStitch {
namespace Testing {
using namespace VideoStitch::Audio;

static const size_t nSamples = 3;

void convUINT8_P() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  auto audio = new uint8_t[nSamples * sizeof(uint8_t)];
  auto dummy = new uint8_t[nSamples * sizeof(uint8_t)];
  raw[0] = audio;
  raw[1] = dummy;

  uint8_t values[nSamples] = {0, 128, 255};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *dummy++ = 0x55;
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::UINT8_P, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::UINT8_P);

  std::cout << "UINT8_P: " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convUINT8() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  auto audio = new uint8_t[nSamples * 2 * sizeof(uint8_t)];
  raw[0] = audio;

  uint8_t values[nSamples] = {0, 128, 255};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *audio++ = 0x55;
  }

  Samples init;
  {
    Samples tmp(SamplingRate::SR_48000, SamplingDepth::UINT8, ChannelLayout::STEREO, 0, raw, nSamples);
    // operator=(Samples&&)
    init = tmp.clone();
  }
  // Samples(Samples&&)
  Samples samples{std::move(init)};

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::UINT8);

  std::cout << "UINT8:   " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convINT16_P() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new uint8_t[nSamples * sizeof(int16_t)];
  raw[1] = new uint8_t[nSamples * sizeof(int16_t)];

  int16_t *audio = (int16_t *)raw[0];
  int16_t *dummy = (int16_t *)raw[1];
  int16_t values[nSamples] = {static_cast<int16_t>(0x8000), 0, 0x7FFF};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *dummy++ = 0x5555;
  }

  Samples init;
  {
    Samples tmp(SamplingRate::SR_48000, SamplingDepth::INT16_P, ChannelLayout::STEREO, 0, raw, nSamples);
    // operator=(Samples&&)
    init = tmp.clone();
  }
  // Samples(Samples&&)
  Samples samples{std::move(init)};

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::INT16_P);

  std::cout << "INT16_P: " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convINT16() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new uint8_t[nSamples * 2 * sizeof(int16_t)];

  int16_t *audio = (int16_t *)raw[0];
  int16_t values[nSamples] = {static_cast<int16_t>(0x8000), 0, 0x7FFF};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *audio++ = 0x5555;
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::INT16, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  Audio::convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::INT16);

  std::cout << "INT16:   " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convINT24_P() {
  const size_t sampleBytes = 3;

  uint8_t *raw[MAX_AUDIO_CHANNELS];
  auto audio = new uint8_t[nSamples * sampleBytes];
  auto dummy = new uint8_t[nSamples * sampleBytes];
  raw[0] = audio;
  raw[1] = dummy;

  uint32_t values[nSamples] = {0x800000, 0, 0x7FFFFF};
  uint32_t dummyVal = 0x555555;
  for (size_t i = 0; i < nSamples; i++) {
    audio[i * sampleBytes] = static_cast<uint8_t>(values[i]);
    audio[i * sampleBytes + 1] = static_cast<uint8_t>(values[i] >> 8);
    audio[i * sampleBytes + 2] = static_cast<uint8_t>(values[i] >> 16);
    dummy[i * sampleBytes] = static_cast<uint8_t>(dummyVal);
    dummy[i * sampleBytes + 1] = static_cast<uint8_t>(dummyVal >> 8);
    dummy[i * sampleBytes + 2] = static_cast<uint8_t>(dummyVal >> 16);
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::INT24_P, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::INT24_P);

  std::cout << "INT24_P: " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convINT24() {
  const size_t sampleBytes = 3;

  uint8_t *raw[MAX_AUDIO_CHANNELS];
  auto audio = new uint8_t[nSamples * 2 * sampleBytes];
  raw[0] = audio;

  uint32_t values[nSamples] = {0x800000, 0, 0x7FFFFF};
  uint32_t dummyVal = 0x555555;
  for (size_t i = 0; i < nSamples; i++) {
    audio[2 * i * sampleBytes] = static_cast<uint8_t>(values[i]);
    audio[2 * i * sampleBytes + 1] = static_cast<uint8_t>(values[i] >> 8);
    audio[2 * i * sampleBytes + 2] = static_cast<uint8_t>(values[i] >> 16);
    audio[2 * i * sampleBytes + 3] = static_cast<uint8_t>(dummyVal);
    audio[2 * i * sampleBytes + 4] = static_cast<uint8_t>(dummyVal >> 8);
    audio[2 * i * sampleBytes + 5] = static_cast<uint8_t>(dummyVal >> 16);
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::INT24, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::INT24);

  std::cout << "INT24:   " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convFLT32_P() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new uint8_t[nSamples * sizeof(float)];
  raw[1] = new uint8_t[nSamples * sizeof(float)];

  float *audio = (float *)raw[0];
  float *dummy = (float *)raw[1];
  float values[nSamples] = {-1.0, 0, 1.0};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *dummy++ = 0.5;
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::FLT_P, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::FLT_P);

  std::cout << "FLT32_P: " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void convFLT32() {
  uint8_t *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new uint8_t[nSamples * 2 * sizeof(float)];

  float *audio = (float *)raw[0];
  float values[nSamples] = {-1, 0, 1};
  for (size_t i = 0; i < nSamples; i++) {
    *audio++ = values[i];
    *audio++ = 0.5;
  }

  Samples samples;
  {
    Samples init(SamplingRate::SR_48000, SamplingDepth::FLT, ChannelLayout::STEREO, 0, raw, nSamples);
    samples = init.clone();
  }

  AudioTrack output(SPEAKER_FRONT_LEFT);
  convertSamplesToMonoDouble(samples, output, 2, SamplingDepth::FLT);

  std::cout << "FLT32:   " << output[0] << ", " << output[1] << ", " << output[2] << std::endl;

  ENSURE_EQ(-1.0, output[0]); /*  -1.0   */
  ENSURE_EQ(0.0, output[1]);  /*   0.0   */
  ENSURE_EQ(1.0, output[2]);  /*   1.0   */
}

void reInitSamples(audioSample_t *s) {
  s[0] = -1.0;
  s[1] = 0.;
  s[2] = 1.0;
}

void convToPlanar() {
  audioSample_t in[3] = {-1.0, 0., 1.0};
  {
    uint8_t *out = reinterpret_cast<uint8_t *>(in);
    int res = convertToSamplesPlanar(in, 3, SamplingDepth::UINT8_P);
    ENSURE_EQ(res, 3, "check number of samples converted");
    ENSURE_EQ(0, (int)out[0], "check minimum value");    // -1.0 -> 0
    ENSURE_EQ(128, (int)out[1], "check value 0");        // 0.0 -> 128
    ENSURE_EQ(255, (int)out[2], "check maximum value");  // 1.0 -> 255
    std::cout << "Test conversion audioSamples to UINT8_P OK" << std::endl;
    reInitSamples(in);
  }

  {
    int16_t *out = reinterpret_cast<int16_t *>(in);
    int res = convertToSamplesPlanar(in, 3, SamplingDepth::INT16_P);
    ENSURE_EQ(res, 3, "check number of samples converted");
    ENSURE_EQ(-32768, (int)out[0], "check minimum value");  // -1.0 -> -32768
    ENSURE_EQ(0, (int)out[1], "check value 0");             // 0.0 -> 0
    ENSURE_EQ(32767, (int)out[2], "check maximum value");   // 1.0 -> 32767
    std::cout << "Test conversion audioSamples to INT16_P OK" << std::endl;
    reInitSamples(in);
  }

  {
    int32_t *out = reinterpret_cast<int32_t *>(in);
    int res = convertToSamplesPlanar(in, 3, SamplingDepth::INT32_P);
    ENSURE_EQ(res, 3, "check number of samples converted");
    ENSURE_EQ(INT32_MIN, (int32_t)out[0], "check minimum value");  // -1.0 -> -2147483648
    ENSURE_EQ(0, (int32_t)out[1], "check value 0");                // 0.0 -> 0
    ENSURE_EQ(INT32_MAX, (int32_t)out[2], "check maximum value");  // 1.0 -> 2147483647
    std::cout << "Test conversion audioSamples to INT32_P OK" << std::endl;
    reInitSamples(in);
  }

  {
      // TODO test INT24_P
  }

  {
    float *out = reinterpret_cast<float *>(in);
    int res = convertToSamplesPlanar(in, 3, SamplingDepth::FLT_P);
    ENSURE_EQ(res, 3, "check number of samples converted");
    ENSURE_EQ((float)-1., out[0], "check minimum value");  // -1.0 -> -1.0
    ENSURE_EQ((float)0., out[1], "check value 0");         // 0.0 -> 0.0
    ENSURE_EQ((float)1., out[2], "check maximum value");   // 1.0 -> 1.0
    std::cout << "Test conversion audioSamples to FLT_P OK" << std::endl;
    reInitSamples(in);
  }

  {
    double *out = reinterpret_cast<double *>(in);
    int res = convertToSamplesPlanar(in, 3, SamplingDepth::DBL_P);
    ENSURE_EQ(res, 3, "check number of samples converted");
    ENSURE_EQ(-1., out[0], "check minimum value");  // -1.0 -> -1.0
    ENSURE_EQ(0., out[1], "check value 0");         // 0.0 -> 0.0
    ENSURE_EQ(1., out[2], "check maximum value");   // 1.0 -> 1.0
    std::cout << "Test conversion audioSamples to DBL_P OK" << std::endl;
    reInitSamples(in);
  }
}

void testConvInterLeavedData() {
  int nSamples = 2;
  audioSample_t **inData = new audioSample_t *[MAX_AUDIO_CHANNELS];
  ChannelLayout inL = STEREO;
  int nChannels = getNbChannelsFromChannelLayout(inL);
  inData[0] = (audioSample_t *)calloc(nSamples, sizeof(audioSample_t));
  inData[1] = (audioSample_t *)calloc(nSamples, sizeof(audioSample_t));
  inData[0][0] = 0.5;
  inData[0][1] = 1.;
  inData[1][0] = -0.5;
  inData[1][1] = -1.;
  // Planar Stereo -> interleave UINT8
  {
    uint8_t *outData = new uint8_t[nSamples * nChannels];
    convertToSamplesInterleaved(inData, nChannels, nSamples, outData, SamplingDepth::UINT8);
    ENSURE_EQ(192, (int)outData[0], "test interleaver");            /* 0.5 */
    ENSURE_EQ(63, (int)outData[1], "test interleaver");             /* -0.5 */
    ENSURE_EQ((int)UINT8_MAX, (int)outData[2], "test interleaver"); /* 1.0 */
    ENSURE_EQ(0, (int)outData[3], "test interleaver");              /* -1.0 */
    std::cout << "test planar stereo -> interleave UINT8 OK" << std::endl;
    delete[] outData;
  }
  // Planar Stereo -> interleave INT16
  {
    int16_t *outData = new int16_t[nSamples * nChannels];
    convertToSamplesInterleaved(inData, nChannels, nSamples, (uint8_t *)outData, SamplingDepth::INT16);
    ENSURE_EQ(INT16_MAX / 2 + 1, (int)outData[0], "test interleaver"); /* 0.5 */
    ENSURE_EQ(INT16_MIN / 2, (int)outData[1], "test interleaver");     /* -0.5 */
    ENSURE_EQ((int)INT16_MAX, (int)outData[2], "test interleaver");    /* 1.0 */
    ENSURE_EQ(INT16_MIN, (int)outData[3], "test interleaver");         /* -1.0 */
    delete[] outData;
    std::cout << "test planar stereo -> interleave INT16 OK" << std::endl;
  }
  // Planar Stereo -> interleave INT24
  {
    uint8_t *outData = new uint8_t[nSamples * nChannels * 3];
    convertToSamplesInterleaved(inData, nChannels, nSamples, (uint8_t *)outData, SamplingDepth::INT24);
    int tmp;
    std::memcpy(&tmp, outData, sizeof(int));
    ENSURE_EQ(INT24_MAX / 2 + 1, tmp, "test interleaver"); /* 0.5 */
    std::memcpy(&tmp, outData + 3, sizeof(int));
    ENSURE_EQ(INT24_MIN / 2, tmp, "test interleaver"); /* -0.5 */
    std::memcpy(&tmp, outData + 6, sizeof(int));
    ENSURE_EQ(INT24_MAX, tmp, "test interleaver"); /* 1.0 */
    delete[] outData;
    std::cout << "test planar stereo -> interleave INT24 OK" << std::endl;
  }
  // Planar Stereo -> interleave INT32
  {
    int32_t *outData = new int32_t[nSamples * nChannels];
    convertToSamplesInterleaved(inData, nChannels, nSamples, (uint8_t *)outData, SamplingDepth::INT32);
    ENSURE_EQ(INT32_MAX / 2 + 1, (int)outData[0], "test interleaver"); /* 0.5 */
    ENSURE_EQ(INT32_MIN / 2, (int)outData[1], "test interleaver");     /* -0.5 */
    ENSURE_EQ(INT32_MAX, (int)outData[2], "test interleaver");         /* 1.0 */
    ENSURE_EQ(INT32_MIN, (int)outData[3], "test interleaver");         /* -1.0 */
    delete[] outData;
    std::cout << "test planar stereo -> interleave INT32 OK" << std::endl;
  }
  // Planar Stereo -> interleave FLT
  {
    float *outData = new float[nSamples * nChannels];
    convertToSamplesInterleaved(inData, nChannels, nSamples, (uint8_t *)outData, SamplingDepth::FLT);
    ENSURE_EQ(outData[0], (float)inData[0][0], "test interleaver");
    ENSURE_EQ(outData[1], (float)inData[1][0], "test interleaver");
    ENSURE_EQ(outData[2], (float)inData[0][1], "test interleaver");
    ENSURE_EQ(outData[3], (float)inData[1][1], "test interleaver");
    delete[] outData;
    std::cout << "test planar stereo -> interleave FLT OK" << std::endl;
  }

  // Planar Stereo -> interleave DBL
  {
    double *outData = new double[nSamples * nChannels];
    convertToSamplesInterleaved(inData, nChannels, nSamples, (uint8_t *)outData, SamplingDepth::DBL);
    ENSURE_EQ(inData[0][0], outData[0], "test interleaver"); /* 0.5 */
    ENSURE_EQ(inData[1][0], outData[1], "test interleaver"); /* -0.5 */
    ENSURE_EQ(inData[0][1], outData[2], "test interleaver"); /* 1.0 */
    ENSURE_EQ(inData[1][1], outData[3], "test interleaver"); /* -1.0 */
    delete[] outData;
    std::cout << "test planar stereo -> interleave DBL OK" << std::endl;
  }
  free(inData[0]);
  free(inData[1]);
  delete[] inData;
}

audioSample_t **callocTestata(int nSamples, ChannelLayout l) {
  audioSample_t **data = new audioSample_t *[MAX_AUDIO_CHANNELS];
  ChannelMap mask = SPEAKER_FRONT_LEFT;
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (mask & l) {
      data[i] = (audioSample_t *)calloc(nSamples, sizeof(audioSample_t));
    } else {
      data[i] = nullptr;
    }
    mask = (ChannelMap)(mask << 1);
  }
  return data;
}
void freeTestata(audioSample_t **data) {
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (data[i]) {
      free(data[i]);
    }
  }
  free(data);
}

audioSample_t **initData(int nSamples, ChannelLayout l) {
  audioSample_t **data = new audioSample_t *[MAX_AUDIO_CHANNELS];
  ChannelMap mask = SPEAKER_FRONT_LEFT;
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (mask & l) {
      data[i] = (audioSample_t *)calloc(nSamples, sizeof(audioSample_t));
      data[i][0] = 1.;
      data[i][1] = 2.;
    } else {
      data[i] = nullptr;
    }
    mask = (ChannelMap)(mask << 1);
  }
  return data;
}

void testMonoConversion(ChannelLayout inL, ChannelLayout outL) {
  AudioBlock inData;
  Samples outData;
  inData.setChannelLayout(inL);
  inData.resize(3);
  inData[SPEAKER_FRONT_LEFT][0] = 1.;
  inData[SPEAKER_FRONT_LEFT][1] = 2.;
  inData[SPEAKER_FRONT_LEFT][2] = 3.;

  AudioResampler *rsp = AudioResampler::create(SamplingRate::SR_22050, SamplingDepth::DBL_P, SamplingRate::SR_22050,
                                               SamplingDepth::DBL_P, outL, 3);
  rsp->resample(inData, outData);

  ENSURE_EQ((int)SamplingDepth::DBL_P, (int)outData.getSamplingDepth(),
            "Check sampling depth");  // just in case the resampler is going crazy
  ENSURE_EQ((int)SamplingRate::SR_22050, (int)outData.getSamplingRate(),
            "Check sampling rate");  // just in case the resampler is going crazy

  ENSURE_EQ(outL, outData.getChannelLayout(), "Check output channel layout");
  ENSURE_EQ(3, (int)outData.getNbOfSamples(), "Check nb samples");

  audioSample_t **dbl = (audioSample_t **)outData.getSamples().data();

  /// Check pan law conversion for MONO to any layout without center channel
  /// In that case, the mono signal is duplicated on the front_left and the front_right channel with a -4.5 dB
  /// attenuation
  ChannelMap m = (ChannelMap)0x1;
  bool hasCenterSpeaker = (((SPEAKER_FRONT_CENTER | SPEAKER_BACK_CENTER) & outL) != 0);
  for (int c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
    m = (ChannelMap)((int64_t)(m) << 1);
    for (int s = 0; s < 3; ++s) {
      if (m & SPEAKER_FRONT_CENTER & outL) {
        // Check front center speaker
        ENSURE_EQ(inData[SPEAKER_FRONT_LEFT][s], dbl[getChannelIndexFromChannelMap(m)][s],
                  "Check front center output samples");
      } else if (m & SPEAKER_BACK_CENTER & outL && !(SPEAKER_FRONT_CENTER & outL)) {
        // Check back center speaker if no front center speaker
        ENSURE_EQ(inData[SPEAKER_FRONT_LEFT][s], dbl[getChannelIndexFromChannelMap(m)][s],
                  "Check back center output samples");
      } else if (m & SPEAKER_FRONT_LEFT && !hasCenterSpeaker) {
        // Check left channel if no center speaker
        ENSURE_EQ(kMonoToStereoNorm * inData[SPEAKER_FRONT_LEFT][s], dbl[getChannelIndexFromChannelMap(m)][s],
                  "Check front left output samples");
      } else if (m & SPEAKER_FRONT_RIGHT && !hasCenterSpeaker) {
        // Check right channel if no center speaker
        ENSURE_EQ(kMonoToStereoNorm * inData[SPEAKER_FRONT_LEFT][s], dbl[getChannelIndexFromChannelMap(m)][s],
                  "Check front right output samples");
      } else if (m & outL) {
        // other speakers should be set to zero
        std::stringstream ss;
        ss << "Check output samples of " << getStringFromChannelType(m);
        ENSURE_EQ((audioSample_t)0, dbl[getChannelIndexFromChannelMap(m)][s], ss.str().c_str());
      }
    }
  }
  delete rsp;
}

void testAnyOtherConversion(ChannelLayout inL, ChannelLayout outL) {
  std::cout << "Test conversion " << getStringFromChannelLayout(inL) << " to " << getStringFromChannelLayout(outL)
            << std::endl;
  ChannelLayout intersect = (ChannelLayout)(inL & outL);

  //  assert( intersect != outL );
  assert(inL != MONO);

  AudioBlock inData;
  Samples outData;
  inData.setChannelLayout(inL);
  inData.resize(3);

  ChannelMap m = (ChannelMap)0x1;
  for (int c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
    if (m & inL) {
      inData[m][0] = 1.;
      inData[m][1] = 2.;
      inData[m][2] = 3.;
    }
    m = (ChannelMap)((int64_t)(m) << 1);
  }

  AudioResampler *rsp = AudioResampler::create(SamplingRate::SR_22050, SamplingDepth::DBL_P, SamplingRate::SR_22050,
                                               SamplingDepth::DBL_P, outL, 3);
  rsp->resample(inData, outData);

  ENSURE_EQ((int)SamplingDepth::DBL_P, (int)outData.getSamplingDepth(),
            "Check sampling depth");  // just in case the resampler is going crazy
  ENSURE_EQ((int)SamplingRate::SR_22050, (int)outData.getSamplingRate(),
            "Check sampling rate");  // just in case the resampler is going crazy

  ENSURE_EQ(outL, outData.getChannelLayout(), "Check output channel layout");
  ENSURE_EQ(3, (int)outData.getNbOfSamples(), "Check nb samples");

  audioSample_t **dbl = (audioSample_t **)outData.getSamples().data();

  m = (ChannelMap)0x1;
  for (int c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
    for (int s = 0; s < 3; ++s) {
      if (m & intersect) {
        ENSURE_EQ(inData[m][s], dbl[getChannelIndexFromChannelMap(m)][s], "Check sample value");
      } else if (m & outL) {
        ENSURE_EQ(0., dbl[getChannelIndexFromChannelMap(m)][s], "Check sample value");
      }
    }
    m = (ChannelMap)((int64_t)(m) << 1);
  }
  delete rsp;
}

/// This test tests only the layout conversion by the resampler
/// That's why we force the sampling rate and the sampling depth to be the same in input and output
/// We check only the output layout and the values of the output samples

void convLayoutSamples() {
  std::cout << "Test MONO -> STEREO" << std::endl;
  testMonoConversion(MONO, STEREO);
  std::cout << "Test MONO -> _2POINT1" << std::endl;
  testMonoConversion(MONO, _2POINT1);
  std::cout << "Test MONO -> 2_1" << std::endl;
  testMonoConversion(MONO, _2_1);
  std::cout << "Test MONO -> SURROUND" << std::endl;
  testMonoConversion(MONO, SURROUND);
  std::cout << "Test MONO -> _3POINT1" << std::endl;
  testMonoConversion(MONO, _3POINT1);
  std::cout << "Test MONO -> _4POINT0" << std::endl;
  testMonoConversion(MONO, _4POINT0);
  std::cout << "Test MONO -> _4POINT1" << std::endl;
  testMonoConversion(MONO, _4POINT1);
  std::cout << "Test MONO -> _2_2" << std::endl;
  testMonoConversion(MONO, _2_2);
  std::cout << "Test MONO -> QUAD" << std::endl;
  testMonoConversion(MONO, QUAD);
  std::cout << "Test MONO -> _5POINT0" << std::endl;
  testMonoConversion(MONO, _5POINT0);
  std::cout << "Test MONO -> _5POINT1" << std::endl;
  testMonoConversion(MONO, _5POINT1);
  std::cout << "Test MONO -> _5POINT0_BACK" << std::endl;
  testMonoConversion(MONO, _5POINT0_BACK);
  std::cout << "Test MONO -> _5POINT1_BACK" << std::endl;
  testMonoConversion(MONO, _5POINT1_BACK);
  std::cout << "Test MONO -> _6POINT0" << std::endl;
  testMonoConversion(MONO, _6POINT0);
  std::cout << "Test MONO -> _6POINT0_FRONT" << std::endl;
  testMonoConversion(MONO, _6POINT0_FRONT);
  std::cout << "Test MONO -> HEXAGONAL" << std::endl;
  testMonoConversion(MONO, HEXAGONAL);
  std::cout << "Test MONO -> _6POINT1" << std::endl;
  testMonoConversion(MONO, _6POINT1);
  std::cout << "Test MONO -> _6POINT1_BACK" << std::endl;
  testMonoConversion(MONO, _6POINT1_BACK);
  std::cout << "Test MONO -> _6POINT1_FRONT" << std::endl;
  testMonoConversion(MONO, _6POINT1_FRONT);
  std::cout << "Test MONO -> _7POINT0" << std::endl;
  testMonoConversion(MONO, _7POINT0);
  std::cout << "Test MONO -> _7POINT0_FRONT" << std::endl;
  testMonoConversion(MONO, _7POINT0_FRONT);
  std::cout << "Test MONO -> _7POINT1" << std::endl;
  testMonoConversion(MONO, _7POINT1);
  std::cout << "Test MONO -> _7POINT1_WIDE" << std::endl;
  testMonoConversion(MONO, _7POINT1_WIDE);
  std::cout << "Test MONO -> _7POINT1_WIDE_BACK" << std::endl;
  testMonoConversion(MONO, _7POINT1_WIDE_BACK);
  std::cout << "Test MONO -> OCTAGONAL" << std::endl;
  testMonoConversion(MONO, OCTAGONAL);

  std::vector<ChannelLayout> layouts = {UNKNOWN,        MONO,
                                        STEREO,         _2POINT1,
                                        _2_1,           SURROUND,
                                        _3POINT1,       _4POINT0,
                                        _4POINT1,       _2_2,
                                        QUAD,           _5POINT0,
                                        _5POINT1,       _5POINT0_BACK,
                                        _5POINT1_BACK,  _6POINT0,
                                        _6POINT0_FRONT, HEXAGONAL,
                                        _6POINT1,       _6POINT1_BACK,
                                        _6POINT1_FRONT, _7POINT0,
                                        _7POINT0_FRONT, _7POINT1,
                                        _7POINT1_WIDE,  _7POINT1_WIDE_BACK,
                                        OCTAGONAL,      AMBISONICS_WXY,
                                        AMBISONICS_WXYZ};

  for (auto inL : layouts) {
    if (inL != UNKNOWN && inL != MONO && inL != AMBISONICS_WXY && inL != AMBISONICS_WXYZ) {
      for (auto outL : layouts) {
        if (outL != UNKNOWN && outL != AMBISONICS_WXY && outL != AMBISONICS_WXYZ) {
          testAnyOtherConversion(inL, outL);
        }
      }
    }
  }
}

///
/// \brief resamplerTest
///        This test the audio resampler
/// \param refPath reference file
/// \param outPath output file
/// \param outrate output sampling rate
///
void resamplerTest(const std::string &refPath, const std::string &outPath, const double outrate) {
  int blockSize = 512;
  // Read input file
  WavReader refFile(refPath);
  refFile.printInfo();
  ChannelLayout layout = MONO;
  if (refFile.getChannels() == 2) {
    layout = STEREO;
  }

  // Create resampler
  AudioResampler *rsp = AudioResampler::create(getSamplingRateFromInt(static_cast<int>(refFile.getSampleRate())),
                                               SamplingDepth::DBL_P, getSamplingRateFromInt(static_cast<int>(outrate)),
                                               SamplingDepth::DBL_P, layout, blockSize);
  ENSURE_EQ(blockSize, rsp->getBlockSize(), "check resampler block size");

  int nSamples = refFile.getnSamples(), nReadSamples = 0;
  int nSamplesToRead = nSamples;
  int nResampled = 0;
  WavWriter outFile(outPath.c_str(), layout, outrate);
  while (nSamplesToRead > 0) {
    AudioBlock inBlock(layout);
    AudioBlock outBlock(layout);
    Samples audioInSamples;
    // read reference file
    if (nSamplesToRead >= blockSize) {
      nReadSamples = blockSize;
    } else if (nReadSamples > 0) {
      nReadSamples = nSamplesToRead;
    } else {
      break;
    }
    refFile.step(inBlock, nReadSamples);
    // Convert AudioBlock to Samples
    audioBlock2Samples(audioInSamples, inBlock);
    // Resample Samples to AudioBlock
    rsp->resample(audioInSamples, outBlock);
    nResampled += outBlock[SPEAKER_FRONT_LEFT].size();
    // Write results in outFile
    outFile.step(outBlock);
    nSamplesToRead -= nReadSamples;
  }
  outFile.close();
  delete rsp;
}

/// Test that there is no crash if the resampler tries to resample to an invalid configuration
void invalidResamplerTest(const std::string &refPath) {
  int blockSize = 512;
  // Read input file
  WavReader refFile(refPath);
  refFile.printInfo();
  ChannelLayout layout = MONO;
  if (refFile.getChannels() == 2) {
    layout = STEREO;
  }

  // Create resampler
  std::vector<std::unique_ptr<AudioResampler>> rsps;
  {
    for (const double outrate : {44100., 48000., 12345., 0., -1.}) {
      for (const SamplingDepth depth : {SamplingDepth::FLT, SamplingDepth::SD_NONE}) {
        AudioResampler *rsp = AudioResampler::create(
            getSamplingRateFromInt(static_cast<int>(refFile.getSampleRate())), SamplingDepth::DBL_P,
            getSamplingRateFromInt(static_cast<int>(outrate)), depth, layout, blockSize);
        ENSURE_EQ(blockSize, rsp->getBlockSize(), "check resampler block size");
        rsps.emplace_back(rsp);
      }
    }
  }

  int nSamples = refFile.getnSamples(), nReadSamples = 0;
  int nSamplesToRead = nSamples;
  while (nSamplesToRead > 0) {
    AudioBlock inBlock(layout);
    Samples audioInSamples;
    // read reference file
    if (nSamplesToRead >= blockSize) {
      nReadSamples = blockSize;
    } else if (nReadSamples > 0) {
      nReadSamples = nSamplesToRead;
    } else {
      break;
    }
    refFile.step(inBlock, nReadSamples);
    // Convert AudioBlock to Samples
    audioBlock2Samples(audioInSamples, inBlock);

    for (const auto &rsp : rsps) {
      AudioBlock outBlock(layout);
      Samples inCopy{audioInSamples.clone()};
      // Resample Samples to AudioBlock
      rsp->resample(inCopy, outBlock);
    }

    nSamplesToRead -= nReadSamples;
  }
}

void testAudioBlockAndInterLeaved() {
  AudioBlock in(_3POINT1);
  in.resize(3);
  in[SPEAKER_FRONT_LEFT][0] = 1;
  in[SPEAKER_FRONT_LEFT][1] = 2;
  in[SPEAKER_FRONT_LEFT][2] = 3;
  in[SPEAKER_FRONT_RIGHT][0] = 4;
  in[SPEAKER_FRONT_RIGHT][1] = 5;
  in[SPEAKER_FRONT_RIGHT][2] = 6;
  in[SPEAKER_FRONT_CENTER][0] = 7;
  in[SPEAKER_FRONT_CENTER][1] = 8;
  in[SPEAKER_FRONT_CENTER][2] = 9;
  in[SPEAKER_LOW_FREQUENCY][0] = 10;
  in[SPEAKER_LOW_FREQUENCY][1] = 11;
  in[SPEAKER_LOW_FREQUENCY][2] = 12;
  std::unique_ptr<audioSample_t[]> out(
      new audioSample_t[in.numSamples() * getNbChannelsFromChannelLayout(in.getLayout())]);
  convertAudioBlockToInterleavedSamples(in, out.get());
  ENSURE_EQ(1., out.get()[0], "Check sample value.");
  ENSURE_EQ(4., out.get()[1], "Check sample value.");
  ENSURE_EQ(7., out.get()[2], "Check sample value.");
  ENSURE_EQ(10., out.get()[3], "Check sample value.");
  ENSURE_EQ(2., out.get()[4], "Check sample value.");
  ENSURE_EQ(5., out.get()[5], "Check sample value.");
  ENSURE_EQ(8., out.get()[6], "Check sample value.");
  ENSURE_EQ(11., out.get()[7], "Check sample value.");
  ENSURE_EQ(3., out.get()[8], "Check sample value.");
  ENSURE_EQ(6., out.get()[9], "Check sample value.");
  ENSURE_EQ(9., out.get()[10], "Check sample value.");
  ENSURE_EQ(12., out.get()[11], "Check sample value.");

  AudioBlock inAgain;
  convertInterleavedSamplesToAudioBlock(out.get(), (int)in.numSamples(), in.getLayout(), inAgain);

  ENSURE(inAgain.getLayout() == in.getLayout(), "Check layout");
  ENSURE(inAgain.numSamples() == in.numSamples(), "Check layout");
  for (const auto &track : inAgain) {
    for (size_t s = 0; s < in.numSamples(); s++) {
      ENSURE(in[track.channel()][s] == track[s], "Check sample value");
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /* argc */, char ** /* argv */) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::convUINT8_P();
  VideoStitch::Testing::convUINT8();

  VideoStitch::Testing::convINT16_P();
  VideoStitch::Testing::convINT16();

  VideoStitch::Testing::convINT24_P();
  VideoStitch::Testing::convINT24();

  VideoStitch::Testing::convFLT32_P();
  VideoStitch::Testing::convFLT32();

  VideoStitch::Testing::convToPlanar();

  VideoStitch::Testing::convLayoutSamples();

  VideoStitch::Testing::testConvInterLeavedData();

  VideoStitch::Testing::testAudioBlockAndInterLeaved();

  // Test resampler mono 44k1 -> 48k
  std::cout << "RUN Test resampler mono 44k1 -> mono 48k" << std::endl;
  std::string testData = VideoStitch::Testing::getDataFolder();
  VideoStitch::Testing::resamplerTest("data/snd/cos44k1.wav", testData + "/rsp.wav", 48000.);
  VideoStitch::Testing::compareWavFile("data/snd/ref_rsp_48k.wav", testData + "/rsp.wav", 0.01, 20);
  std::cout << "RUN Test resampler mono 44k1 -> mono 48k : PASSED" << std::endl;

  // Test resampler stereo 48k -> stereo 44k1
  std::cout << "RUN Test resampler stereo 48k -> stereo 44k1" << std::endl;
  VideoStitch::Testing::resamplerTest("data/snd/test_rsp_48k_stereo.wav", testData + "/rsp_stereo.wav", 44100.);
  VideoStitch::Testing::compareWavFile("data/snd/ref_rsp_cos44k1_stereo.wav", testData + "/rsp_stereo.wav",
                                       10. / 32267., 20);
  std::cout << "Test resampler stereo 48k -> stereo 44k1 : PASSED" << std::endl;

  VideoStitch::Testing::invalidResamplerTest("data/snd/ref_rsp_cos44k1_stereo.wav");

  return 0;
}
