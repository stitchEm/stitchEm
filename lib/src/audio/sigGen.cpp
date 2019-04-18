// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sigGen.hpp"

#include "resampler.hpp"

#include <vector>
#include <sstream>

namespace VideoStitch {
namespace Audio {

///////////////////////////////////////////////////////////
//
//               SigGenSine1Dim
//
///////////////////////////////////////////////////////////
//
// Generate sine wave samples for any sample rate
// or frequency for one dimension only
//
//
SigGenSine::SigGenSine1Dim::SigGenSine1Dim(double freq, double sampleRate, double amp)
    : _sampleRate(sampleRate), _freq(freq), _amp(amp), _readPos(0) {
  _nSamplesInWave = (unsigned int)(sampleRate * (1 / _freq));
  _samples = (audioSample_t*)malloc((size_t)_nSamplesInWave * sizeof(audioSample_t));
  _makeSamples();
}

SigGenSine::SigGenSine1Dim::SigGenSine1Dim(const SigGenSine1Dim& s)
    : _sampleRate(s._sampleRate),
      _freq(s._freq),
      _amp(s._amp),
      _nSamplesInWave(s._nSamplesInWave),
      _samples((audioSample_t*)malloc((size_t)_nSamplesInWave * sizeof(audioSample_t))),
      _readPos(s._readPos) {
  memcpy(_samples, s._samples, _nSamplesInWave * sizeof(audioSample_t));
}

SigGenSine::SigGenSine1Dim::~SigGenSine1Dim() { free(_samples); }

void SigGenSine::SigGenSine1Dim::_makeSamples() {
  for (unsigned int i = 0; i < _nSamplesInWave; i++) {
    _samples[i] = (audioSample_t)_amp * cos((2 * M_PI) * ((double)i / (double)_nSamplesInWave));
  }
}

void SigGenSine::SigGenSine1Dim::step(AudioTrack& buf) {
  size_t nSamples = buf.size();
  for (size_t i = 0; i < nSamples; i++) {
    buf[i] = (audioSample_t)_samples[_readPos++ % _nSamplesInWave];
  }
}

double SigGenSine::SigGenSine1Dim::getFrequency(void) const { return _freq; }

void SigGenSine::SigGenSine1Dim::setFrequency(double freq) {
  // Save current sample value
  audioSample_t curSample = _samples[_readPos % _nSamplesInWave];

  // Save the direction the wave was going true = up, false = down
  bool dir = _samples[_readPos % _nSamplesInWave] <= _samples[(_readPos + 1) % _nSamplesInWave];

  // Re-generate samples
  _freq = freq;
  free(_samples);
  _nSamplesInWave = (unsigned int)(_sampleRate * (1 / _freq));
  _samples = (audioSample_t*)malloc((size_t)_nSamplesInWave * sizeof(audioSample_t));
  _makeSamples();

  // Find the sample in the new waveform that is closest to the previous one,
  // with the wave going the same direction (up or down).
  _readPos = 0;
  audioSample_t diff = 3;  // Biggest possible difference is 2

  for (unsigned int i = 0; i < _nSamplesInWave; i++) {
    if (dir) {  // Going up
      if (((_samples[i] - curSample) >= 0) && ((_samples[i] - curSample) < diff)) {
        if (_samples[i] > _samples[(i + 1) % _nSamplesInWave]) {
          continue;
        }
        diff = _samples[i] - curSample;
        _readPos = i;
      }
    } else {  // Going down
      if (((curSample - _samples[i]) >= 0) && ((curSample - _samples[i]) < diff)) {
        if (_samples[i] < _samples[(i + 1) % _nSamplesInWave]) {
          continue;
        }
        diff = curSample - _samples[i];
        _readPos = i;
      }
    }
  }
}

///////////////////////////////////////////////////////////
//
//               Input::AudioReader::SigGenSine
//
///////////////////////////////////////////////////////////
//
// Adapter from SigGenSine to AudioReader
//
//

SigGenSineInput::SigGenSineInput(readerid_t id, SamplingRate sampleRate, ChannelLayout layout,
                                 std::vector<double>& freqs, double amp)
    : Reader(id),
      Input::AudioReader(layout, sampleRate, SamplingDepth::DBL_P),
      _impl(freqs, (double)getIntFromSamplingRate(sampleRate), amp),
      _timestamp(0) {}

SigGenSineInput::~SigGenSineInput() {}

Input::ReadStatus SigGenSineInput::readSamples(size_t nbSamples, Samples& audioSamples) {
  AudioBlock block(getSpec().layout);
  for (auto& track : block) {
    for (size_t s = 0; s < nbSamples; s++) {
      track.push_back(0);
    }
  }
  _impl.step(block);
  audioBlock2Samples(audioSamples, block);

  return Input::ReadStatus::OK();
}

Status SigGenSineInput::seekFrame(mtime_t) { return Status::OK(); }

size_t SigGenSineInput::available() { return 1024; }

bool SigGenSineInput::eos() { return false; }

}  // namespace Audio
}  // namespace VideoStitch
