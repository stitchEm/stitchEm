// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// A class to resample audio samples

#pragma once

#include "libvideostitch/status.hpp"
#include "libvideostitch/audio.hpp"
#include "libvideostitch/audioObject.hpp"
#include "converter.hpp"
#include <fstream>

#ifndef R8BLIB_UNSUPPORTED
#if _MSC_VER
// To disable warnings on the external r8b library
#pragma warning(push)
#pragma warning(disable : 4127)
#pragma warning(disable : 4201)
#include <r8b-src/CDSPResampler.h>
#pragma warning(pop)
#else
#include <r8b-src/CDSPResampler.h>
#endif
#endif

namespace VideoStitch {
namespace Audio {

// Helper to convert Samples to AudioBlock
void samples2AudioBlock(AudioBlock &out, const Samples &in);

// Helper to convert AudioBlock to Samples
void audioBlock2Samples(Samples &out, const AudioBlock &in);

class AudioResampler {
 public:
  ~AudioResampler();

  ///
  /// \brief Create an audio resampler
  /// \param _inRate    input sampling rate
  /// \param _inDepth   input sampling depth
  /// \param _outRate   output sampling rate
  /// \param _outDepth  output sampling depth
  /// \param _layout    audio layout
  /// \param _blockSize size in samples of the internal block
  ///
  static AudioResampler *create(const SamplingRate _inRate, const SamplingDepth _inDepth, const SamplingRate _outRate,
                                const SamplingDepth _outDepth, const ChannelLayout _layout, const size_t _blockSizeIn);

  ///
  /// \brief Resamples the input audio to output audio
  ///        the input should correspond to the channel index given
  ///        C-style function the memory has to be managed by the user
  /// \param in   Pointer on the input buffer
  /// \param nbSamplesin   number of input samples
  /// \param out  Pointer on the output buffer
  /// \param channelIndex Index of the channel corresponding to the input pointer
  ///
  int resample(const audioSample_t *in, size_t nbSamplesin, audioSample_t *&out, const uint32_t channelIndex);

  ///
  /// \brief Resamples the input audio to output audio
  ///        This transforms an Audio::Samples and into an AudioBloack
  ///
  /// \param audioSamplesIn   input audio samples
  /// \param audioBlockOut    output audio samples
  ///
  void resample(const Audio::Samples &audioSamplesIn, AudioBlock &audioBlockOut);

  ///
  /// \brief Resamples the input audio to output audio
  ///        This transforms an AudioBlock into an Audio::Samples
  ///
  /// \param audioBlockIn     input audio samples
  /// \param audioSamplesOut  output audio samples
  ///
  void resample(const AudioBlock &audioBlockIn, Audio::Samples &audioSamplesOut);

  ///
  /// \brief Returns the blockSize
  /// \return block size
  ///
  int getBlockSize() { return _blockSizeIn; }

 private:
  ///
  /// \brief Constructs an audio resampler
  /// \param _inRate    input sampling rate
  /// \param _inDepth   input sampling depth (not used)
  /// \param _outRate   output sampling rate
  /// \param _outDepth  output sampling depth (not used)
  /// \param _layout    audio layout
  /// \param _blockSize size in samples of the internal block
  ///
  AudioResampler(const SamplingRate _inRate, const SamplingDepth, const SamplingRate _outRate, const SamplingDepth,
                 const ChannelLayout _layout, const int _blockSizeIn);
  int convertFltPToDblP(const float *in, audioSample_t *out);
  void alloc();
  void dumpInput(const ChannelMap channelType);
  void dumpOutput(int nResampled, const int iChannel, const ChannelMap channelType);
#ifndef R8BLIB_UNSUPPORTED
  r8b::CPtrKeeper<r8b::CDSPResampler24 *> _resamps[MAX_AUDIO_CHANNELS];
#endif
  mtime_t _offsettime;
  double _inRate;   // source sample rate
  double _outRate;  // destination sample rate
  SamplingDepth _inDepth;
  SamplingDepth _outDepth;
  ChannelLayout _layout;
  int _blockSizeIn;
  int _blockSizeOut;
#ifndef R8BLIB_UNSUPPORTED
  r8b::CFixedBuffer<audioSample_t> _internalBuf;
#endif
  audioSample_t **_outData;
  audioSample_t **_keepPtr;
  bool _dump;
  std::ofstream _inFile;
  std::ofstream _outFile;
  int _iWriteCount;
};

}  // namespace Audio
}  // namespace VideoStitch
