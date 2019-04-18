// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Asynchronous rate conveter
//
// Wrapper of Audio::Samples and Resampler to read from different audio reader
// Convert everything in the internal format of AudioBlock

#pragma once

#include "resampler.hpp"

#include "libvideostitch/audioBlock.hpp"
#include "libvideostitch/input.hpp"

#include <queue>

namespace VideoStitch {
namespace Audio {

class AudioAsyncReader {
 public:
  explicit AudioAsyncReader(std::shared_ptr<Input::AudioReader> audioreader, const BlockSize bs, const SamplingRate sr);
  ~AudioAsyncReader();
  static AudioAsyncReader* create(std::shared_ptr<Input::AudioReader> audioreader, const BlockSize blockSize,
                                  const SamplingRate internalSr);
  std::shared_ptr<Input::AudioReader> getDelegate() const { return delegate; }

  Input::ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) {
    return delegate->readSamples(rescale(nbSamples), audioSamples);
  }

  bool eos() { return delegate->eos(); }

  size_t rescale(size_t nSamples) {
    double readerSr = static_cast<double>(Audio::getIntFromSamplingRate(delegate->getSpec().sampleRate));
    return static_cast<size_t>(nSamples * readerSr / internalSamplingRate);
  }

  size_t available() { return rescale(delegate->available()); }

  readerid_t getId() const { return delegate->id; }

  const Input::AudioReader::Spec& getSpec() const { return delegate->getSpec(); }

  void resample(const Samples& samplesIn, AudioBlock& blockOut);

 private:
  std::shared_ptr<Input::AudioReader> delegate;
  double internalSamplingRate;
  AudioResampler* rsp;
};

}  // namespace Audio
}  // namespace VideoStitch
