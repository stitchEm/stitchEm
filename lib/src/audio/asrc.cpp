// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Asynchronous rate conveter
//
// Wrapper of Audio::Samples and Resampler to read from different audio reader
// Convert everything in the internal format of AudioBlock

#include "asrc.hpp"

namespace VideoStitch {
namespace Audio {

AudioAsyncReader::AudioAsyncReader(std::shared_ptr<Input::AudioReader> audioreader, const BlockSize bs,
                                   const SamplingRate internalSr)
    : delegate(audioreader), internalSamplingRate(static_cast<double>(Audio::getIntFromSamplingRate(internalSr))) {
  Input::AudioReader::Spec spec = delegate->getSpec();
  size_t blockSizeIn = static_cast<size_t>(getDblFromBlockSize(bs) * getDblFromSamplingRate(spec.sampleRate) /
                                           getDblFromSamplingRate(internalSr));
  rsp = AudioResampler::create(spec.sampleRate, spec.sampleDepth, internalSr, SamplingDepth::DBL_P, spec.layout,
                               blockSizeIn);
}

AudioAsyncReader *AudioAsyncReader::create(std::shared_ptr<Input::AudioReader> audioreader, const BlockSize bs,
                                           const SamplingRate internalSr) {
  return new AudioAsyncReader(audioreader, bs, internalSr);
}

AudioAsyncReader::~AudioAsyncReader() { delete rsp; }

void AudioAsyncReader::resample(const Samples &samplesIn, AudioBlock &blockOut) { rsp->resample(samplesIn, blockOut); }

}  // namespace Audio
}  // namespace VideoStitch
