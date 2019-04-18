// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Class to convert audio samples from DOUBLE PLANAR to any requested format

#pragma once

#include "libvideostitch/audioObject.hpp"
#include "libvideostitch/audio.hpp"

namespace VideoStitch {
namespace Audio {

#define INT24_MAX (8388607)
#define INT24_MIN (-8388607 - 1)

static const double kMonoToStereoNorm = 0.59566214352;  // pan law (corresponds to -4.5 dB)

///
/// \fn    convertSamplesToMonoDouble
/// \brief Converts channel 0 of samples to double and places them in snd
///
/// \param samples The samples to be converted
/// \param snd The output vector of samples converted to double
/// \param nChannels The number of channels in the input
/// \param sampleDepth The format of the input samples
///
void convertSamplesToMonoDouble(const Audio::Samples &samples, Audio::AudioTrack &snd, const int nChannels,
                                const Audio::SamplingDepth sampleDepth);

int convertSamples(const std::vector<std::vector<audioSample_t> > &in, uint8_t *outData, const SamplingDepth outDepth);

int convertToSamples(const audioSample_t **in, size_t nChannels, size_t nSamples, uint8_t *outData,
                     const SamplingDepth outDepth);

// 1-dim converter
int convertToSamplesPlanar(audioSample_t *inout, size_t nSamples, const SamplingDepth outDepth);

// Multi-channel converter for interleaved data
int convertToSamplesInterleaved(audioSample_t **inData, size_t nChannels, size_t nSamples, uint8_t *outData,
                                SamplingDepth outDepth);

int convertToInternalFormat(const uint8_t *in, const int inSize, const SamplingDepth inDepth, audioSample_t *out);

///
/// \fn    convertInterleaveData
/// \brief deinterleave one channel to audiosample_t
///
/// \param in pointer to input data
/// \param inSize number of samples per channel
/// \param inDepth input sampling depth
/// \param inLayout input channel layout
/// \param channelIndex channel index
/// \param out pointer on the output data (memory has to be managed by the user)
///
int convertInterleaveData(const uint8_t *in, const int inSize, const SamplingDepth inDepth,
                          const ChannelLayout inLayout, const int channelIndex, audioSample_t *out);

size_t getSamplingDepthSize(SamplingDepth depth);

bool isInterleaved(const SamplingDepth depth);

void convertToLayout(uint8_t **inData, uint8_t **outData, int nSamples, SamplingDepth sd, ChannelLayout inLayout,
                     ChannelLayout outLayout);

template <typename T>
void convMonoToStereo(T **inData, T **outData, int nSamples, T norm) {
  for (int i = 0; i < nSamples; ++i) {
    outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_LEFT)][i] = norm * inData[0][i];
    outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_RIGHT)][i] =
        outData[getChannelIndexFromChannelMap(SPEAKER_FRONT_LEFT)][i];
  }
}

/// \fn convertAudioBlockToInterleavedSamples
/// \brief Convert data from audio block to an array of interleaved audioSample_t
/// \param in Input AudioBlock
/// \param out Pointer to the output array (memory has to be allocated before
///            calling this function)
void convertAudioBlockToInterleavedSamples(const AudioBlock &in, audioSample_t *const out);

/// \fn convertInterleavedSamplesToAudioBlock
/// \brief Convert interleaved data to an audioBlock.
/// \param in Pointer on the input data.
/// \param nSamples Number of samples per channel.
/// \param layout Channel layout of audio (stereo, 5.1, etc.).
/// \param out Output AudioBlock.
void convertInterleavedSamplesToAudioBlock(const audioSample_t *const in, int nSamples, ChannelLayout layout,
                                           AudioBlock &outBlock);

/// \fn convertAudioBlockToPlanarSamples
/// \brief Convert data from audio block to a 2D array of planar audioSample_t
/// \param in input AudioBlock
/// \param out Pointer to the output array (memory has to be allocated before
///            calling this function)
void convertAudioBlockToPlanarSamples(const AudioBlock &in, audioSample_t *const *const out);

/// \fn convertPlanarSamplesToAudioBlock
/// \brief Convert planar audio data to an audioBlock.
/// \param in Pointer to the input data.
/// \param nSamples Number of samples per channel.
/// \param layout Channel layout of audio (stereo, 5.1, etc.).
/// \param out Output AudioBlock.
void convertPlanarSamplesToAudioBlock(const audioSample_t *const *const in, int nSamples, ChannelLayout layout,
                                      AudioBlock &outBlock);

}  // namespace Audio
}  // namespace VideoStitch
