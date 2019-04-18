// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "converter.hpp"

#include "libvideostitch/logging.hpp"

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>

#define INT24SIZE 3

namespace VideoStitch {
namespace Audio {

size_t getSamplingDepthSize(SamplingDepth depth) {
  switch (depth) {
    case SamplingDepth::UINT8:
    case SamplingDepth::UINT8_P:
      return sizeof(uint8_t);
    case SamplingDepth::INT16:
    case SamplingDepth::INT16_P:
      return sizeof(int16_t);
    case SamplingDepth::INT24:
    case SamplingDepth::INT24_P:
      return sizeof(int8_t) * 3;
    case SamplingDepth::INT32:
    case SamplingDepth::INT32_P:
      return sizeof(int32_t);
    case SamplingDepth::FLT:
    case SamplingDepth::FLT_P:
      return sizeof(float);
    case SamplingDepth::DBL:
    case SamplingDepth::DBL_P:
      return sizeof(double);
    case SamplingDepth::SD_NONE:
      return 0;
  }
  return 0;
}

// Convert 1-dim sample array for planar format from double to any planar output format
int convertToSamplesPlanar(audioSample_t *inout, size_t nSamples, const SamplingDepth outDepth) {
  switch (outDepth) {
    case SamplingDepth::UINT8_P: {
      uint8_t *out8 = reinterpret_cast<uint8_t *>(inout);
      for (size_t s = 0; s < nSamples; s++) {
        audioSample_t x = inout[s];
        if (x > 0) {
          out8[s] = (uint8_t)((x * 127. + 128.));
        } else {
          out8[s] = (uint8_t)((x * 128. + 128.));
        }
      }
    } break;

    case SamplingDepth::INT16_P: {
      int16_t *out16 = reinterpret_cast<int16_t *>(inout);
      for (size_t s = 0; s < nSamples; s++) {
        if (inout[s] > 0) {
          out16[s] = (int16_t)(inout[s] * (INT16_MAX + 0.5));
        } else {
          out16[s] = (int16_t)(inout[s] * (-INT16_MIN));
        }
      }
    } break;

    case SamplingDepth::INT24_P: {
      uint8_t *out24 = reinterpret_cast<uint8_t *>(inout);
      for (size_t s = 0; s < nSamples; s++) {
        audioSample_t x = inout[s];
        int32_t y;
        uint8_t *yy = (uint8_t *)&y;
        if (x > 0) {
          y = (int32_t)(x * INT24_MAX + 0.5);
        } else {
          y = (int32_t)(x * (-INT24_MIN));
        }
        out24[s * INT24SIZE] = yy[0];
        out24[s * INT24SIZE + 1] = yy[1];
        out24[s * INT24SIZE + 2] = yy[2];
      }
    } break;

    case SamplingDepth::INT32_P: {
      int32_t *out32 = reinterpret_cast<int32_t *>(inout);
      for (size_t s = 0; s < nSamples; s++) {
        if (inout[s] > 0) {
          out32[s] = (int32_t)(inout[s] * (INT32_MAX) + 0.5);
        } else {
          out32[s] = (int32_t)(inout[s] * (-(double)INT32_MIN));
        }
      }
    } break;

    case SamplingDepth::FLT_P: {
      float *outflt = reinterpret_cast<float *>(inout);
      for (size_t s = 0; s < nSamples; s++) {
        outflt[s] = (float)inout[s];
      }
    } break;

    case SamplingDepth::DBL_P: {
      // Nothing to do as it is already the internal format
    } break;
    default:
      return 0;
  }
  return static_cast<int>(nSamples);
}

// convert samples to interleaved data
int convertToSamplesInterleaved(audioSample_t **inData, size_t nChannels, size_t nSamples, uint8_t *outData,
                                SamplingDepth outDepth) {
  switch (outDepth) {
    case SamplingDepth::UINT8:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          audioSample_t x = inData[c][s];
          if (x > 0) {
            x = std::min(x, 1.0);
            outData[s * nChannels + c] = (uint8_t)((x * 127. + 128.) + 0.5);
          } else {
            x = std::max(x, -1.0);
            outData[s * nChannels + c] = (uint8_t)((x * 128. + 128.) - 0.5);
          }
        }
      }
      break;

    case SamplingDepth::INT16:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          audioSample_t x = inData[c][s];
          if (x > 0) {
            x = std::min(x, 1.0);
            reinterpret_cast<int16_t *>(outData)[s * nChannels + c] = (int16_t)(x * (INT16_MAX) + 0.5);
          } else {
            x = std::max(x, -1.0);
            reinterpret_cast<int16_t *>(outData)[s * nChannels + c] = (int16_t)(x * (-INT16_MIN) - 0.5);
          }
        }
      }
      break;

    case SamplingDepth::INT24:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          uint8_t *out = outData + (s * nChannels + c) * INT24SIZE;
          audioSample_t x = inData[c][s];
          int32_t y = 0;
          uint8_t *yy = (uint8_t *)&y;
          if (x > 0) {
            x = std::min(x, 1.0);
            y = (int32_t)(x * (INT24_MAX) + 0.5);
          } else {
            x = std::max(x, -1.0);
            y = (int32_t)(x * (-INT24_MIN) - 0.5);
          }
          out[0] = yy[0];
          out[1] = yy[1];
          out[2] = yy[2];
        }
      }
      break;

    case SamplingDepth::INT32:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          audioSample_t x = inData[c][s];
          if (x > 0) {
            x = std::min(x, 1.0);
            reinterpret_cast<int32_t *>(outData)[s * nChannels + c] = (int32_t)(x * (INT32_MAX) + 0.5);
          } else {
            x = std::max(x, -1.0);
            reinterpret_cast<int32_t *>(outData)[s * nChannels + c] = (int32_t)(x * (-(double)INT32_MIN) - 0.5);
          }
        }
      }
      break;

    case SamplingDepth::FLT:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          reinterpret_cast<float *>(outData)[s * nChannels + c] = (float)inData[c][s];
        }
      }
      break;

    case SamplingDepth::DBL:
      for (size_t s = 0; s < nSamples; s++) {
        for (size_t c = 0; c < nChannels; c++) {
          reinterpret_cast<double *>(outData)[s * nChannels + c] = inData[c][s];
        }
      }
      break;

    default:
      return 0;
  }
  return static_cast<int>(nSamples);
}

// convert a 1-dim array from any depth to an audioSample_t array
// only planar formats
int convertToInternalFormat(const uint8_t *in, const int inSize, const SamplingDepth inDepth, audioSample_t *out) {
  switch (inDepth) {
    case SamplingDepth::UINT8_P:
      for (int i = 0; i < inSize; i++) {
        if (out[i] > 0) {
          out[i] = (static_cast<audioSample_t>(in[i]) - 128.0) / 127.0;
        } else {
          out[i] = (static_cast<audioSample_t>(in[i]) - 128.0) / 128.0;
        }
      }
      break;
    case SamplingDepth::INT16_P: {
      const int16_t *pint16 = reinterpret_cast<const int16_t *>(in);
      for (int i = 0; i < inSize; i++) {
        if (pint16[i] > 0) {
          out[i] = static_cast<audioSample_t>(pint16[i]) / INT16_MAX;
        } else {
          out[i] = static_cast<audioSample_t>(pint16[i]) / (INT16_MAX + 1);
        }
      }
      break;
    }
    case SamplingDepth::INT24_P:
      /// Not Managed
      Logger::get(Logger::Error) << "Input format not managed " << getStringFromSamplingDepth(inDepth) << std::endl;
      return -1;
    case SamplingDepth::INT32_P: {
      const int32_t *pint32 = reinterpret_cast<const int32_t *>(in);
      for (int i = 0; i < inSize; i++) {
        if (pint32[i] > 0) {
          out[i] = static_cast<audioSample_t>(pint32[i]) / INT32_MAX;
        } else {
          out[i] = static_cast<audioSample_t>(pint32[i]) / static_cast<audioSample_t>(INT32_MAX + 1.);
        }
      }
      break;
    }
    case SamplingDepth::FLT_P: {
      const float *pfloat = reinterpret_cast<const float *>(in);
      for (int i = 0; i < inSize; i++) {
        out[i] = static_cast<audioSample_t>(pfloat[i]);
      }
      break;
    }
    case SamplingDepth::DBL_P: {
      const audioSample_t *pdbl = reinterpret_cast<const audioSample_t *>(in);
      for (int i = 0; i < inSize; i++) {
        out[i] = pdbl[i];
      }
      break;
    }
    default:
      Logger::get(Logger::Error) << "Wrong input depth format (should be planar) "
                                 << getStringFromSamplingDepth(inDepth) << std::endl;
      return -1;
  }

  return inSize;
}

// convert a 1-dim array from any depth to an audioSample_t array
// only interleaved formats
int convertInterleaveData(const uint8_t *in, const int inSize, const SamplingDepth inDepth,
                          const ChannelLayout inLayout, const int channelIndex, audioSample_t *out) {
  int nbchannels = getNbChannelsFromChannelLayout(inLayout);
  switch (inDepth) {
    case SamplingDepth::UINT8:
      for (int i = 0; i < inSize; i++) {
        if (out[i] > 0) {
          out[i] = (static_cast<audioSample_t>(in[i * nbchannels + channelIndex]) - 128.0) / 127.0;
        } else {
          out[i] = (static_cast<audioSample_t>(in[i * nbchannels + channelIndex]) - 128.0) / 128.0;
        }
      }
      break;
    case SamplingDepth::INT16: {
      const int16_t *pint16 = reinterpret_cast<const int16_t *>(in);
      for (int i = 0; i < inSize; i++) {
        if (pint16[i] > 0) {
          out[i] = static_cast<audioSample_t>(pint16[i * nbchannels + channelIndex]) / INT16_MAX;
        } else {
          out[i] = static_cast<audioSample_t>(pint16[i * nbchannels + channelIndex]) / (INT16_MAX + 1);
        }
      }
      break;
    }
    case SamplingDepth::INT24:
      /// Not Managed
      return -1;
    case SamplingDepth::INT32: {
      const int32_t *pint32 = reinterpret_cast<const int32_t *>(in);
      for (int i = 0; i < inSize; i++) {
        if (pint32[i] > 0) {
          out[i] = static_cast<audioSample_t>(pint32[i * nbchannels + channelIndex]) / INT32_MAX;
        } else {
          out[i] = static_cast<audioSample_t>(pint32[i * nbchannels + channelIndex]) /
                   static_cast<audioSample_t>(INT32_MAX + 1.);
        }
      }
      break;
    }
    case SamplingDepth::FLT: {
      const float *pfloat = reinterpret_cast<const float *>(in);
      for (int i = 0; i < inSize; i++) {
        out[i] = static_cast<audioSample_t>(pfloat[i * nbchannels + channelIndex]);
      }
      break;
    }
    case SamplingDepth::DBL: {
      const audioSample_t *pdbl = reinterpret_cast<const audioSample_t *>(in);
      for (int i = 0; i < inSize; i++) {
        out[i] = pdbl[i * nbchannels + channelIndex];
      }
      break;
    }
    default:
      Logger::get(Logger::Error) << "Wrong input depth format (should be interleaved) "
                                 << getStringFromSamplingDepth(inDepth) << std::endl;
      return -1;
  }
  return inSize;
}

bool isInterleaved(const SamplingDepth depth) {
  bool isInterleaved = false;
  switch (depth) {
    case SamplingDepth::UINT8:
    case SamplingDepth::INT16:
    case SamplingDepth::INT24:
    case SamplingDepth::INT32:
    case SamplingDepth::FLT:
    case SamplingDepth::DBL:
      isInterleaved = true;
      break;
    case SamplingDepth::UINT8_P:
    case SamplingDepth::INT16_P:
    case SamplingDepth::INT24_P:
    case SamplingDepth::INT32_P:
    case SamplingDepth::FLT_P:
    case SamplingDepth::DBL_P:
      isInterleaved = false;
      break;
    case SamplingDepth::SD_NONE:
      Logger::get(Logger::Error) << "Error sampling depth not initialized (SD_NONE)" << std::endl;
      isInterleaved = false;
      break;
  }
  return isInterleaved;
}

void convMonoToStereo(uint8_t **inData, uint8_t **outData, int nSamples, SamplingDepth sd) {
  if (sd == SamplingDepth::DBL_P) {
    double **outDbl = (double **)outData;
    double **inDbl = (double **)inData;
    convMonoToStereo(inDbl, outDbl, nSamples, kMonoToStereoNorm);
  } else if (sd == SamplingDepth::FLT_P) {
    float **out = (float **)outData;
    float **in = (float **)inData;
    convMonoToStereo(in, out, nSamples, (float)kMonoToStereoNorm);
  } else if (sd == SamplingDepth::INT32_P || sd == SamplingDepth::INT24_P || sd == SamplingDepth::INT16_P ||
             sd == SamplingDepth::UINT8_P) {
    // TODO add pan law: for the moment copy duplicate on each channel
    // the pan law conversion should be performed when data are in double
    memcpy(outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_LEFT)], inData[0],
           nSamples * getSampleSizeFromSamplingDepth(sd));
    memcpy(outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_RIGHT)], inData[0],
           nSamples * getSampleSizeFromSamplingDepth(sd));
  }
}

// Convert data to the output layout requested (input and output must be planar)
void convertToLayout(uint8_t **inData, uint8_t **outData, int nSamples, SamplingDepth sd, ChannelLayout inLayout,
                     ChannelLayout outLayout) {
  assert(isInterleaved(sd) == false);
  if (inLayout == MONO) {
    // Copy data to the center speaker if one is present
    if (outLayout & SPEAKER_FRONT_CENTER) {
      memcpy(outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_CENTER)], inData[0],
             nSamples * getSampleSizeFromSamplingDepth(sd));
    } else if (outLayout & SPEAKER_BACK_CENTER) {
      memcpy(outData[getChannelIndexFromChannelMap(SPEAKER_BACK_CENTER)], inData[0],
             nSamples * getSampleSizeFromSamplingDepth(sd));
    } else {
      // if no center speaker just convert it to stereo data on the front left and right left speaker
      convMonoToStereo(inData, outData, nSamples, sd);
    }
  } else {
    for (int c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
      if (getChannelMapFromChannelIndex(c) & outLayout & inLayout) {
        // copy data if the channel is present in the input and output
        memcpy(outData[c], inData[c], nSamples * getSampleSizeFromSamplingDepth(sd));
      } else if (getChannelMapFromChannelIndex(c) & outLayout) {
        // Sets the output channel to 0 if it is not present in the input
        memset(outData[c], 0, nSamples * getSampleSizeFromSamplingDepth(sd));
      }
    }
  }
}

void convertAudioBlockToInterleavedSamples(const AudioBlock &in, audioSample_t *const out) {
  size_t numChannels = getNbChannelsFromChannelLayout(in.getLayout());
  for (size_t i = 0; i < in.numSamples(); ++i) {
    size_t c = 0;
    for (const auto &track : in) {
      out[i * numChannels + c] = track[i];
      c++;
    }
  }
}

void convertInterleavedSamplesToAudioBlock(const audioSample_t *const in, int nSamples, ChannelLayout layout,
                                           AudioBlock &outBlock) {
  outBlock.setChannelLayout(layout);
  outBlock.resize(nSamples);

  int c = 0, nChannels = getNbChannelsFromChannelLayout(layout);
  for (auto &track : outBlock) {
    for (int s = 0; s < nSamples; s++) {
      track[s] = in[s * nChannels + c];
    }
    c++;
  }
}

void convertAudioBlockToPlanarSamples(const AudioBlock &in, audioSample_t *const *const out) {
  size_t c = 0;
  for (const auto &track : in) {
    std::copy(track.begin(), track.end(), out[c++]);
  }
}

void convertPlanarSamplesToAudioBlock(const audioSample_t *const *const in, int nSamples, ChannelLayout layout,
                                      AudioBlock &outBlock) {
  outBlock.setChannelLayout(layout);
  outBlock.resize(nSamples);
  int c = 0;
  for (auto &track : outBlock) {
    std::copy(in[c], in[c] + nSamples, track.data());
    c++;
  }
}

}  // namespace Audio
}  // namespace VideoStitch
