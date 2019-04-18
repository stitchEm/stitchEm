// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audioObject.hpp"
#include "libvideostitch/input.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace VideoStitch {
namespace Audio {

/// Sine generator for multiple channels
/// wrap one SigGenSine1Dim by channel
class SigGenSine : public AudioObject {
 public:
  /// Sine generator for 1 dimension only
  class SigGenSine1Dim {
   public:
    SigGenSine1Dim(double freq, double sampleRate = getDefaultSamplingRate(), double amp = 1.);
    SigGenSine1Dim(const SigGenSine1Dim &s);
    ~SigGenSine1Dim();
    double getFrequency(void) const;
    void setFrequency(double freq);
    void step(AudioTrack &buf);  // one channel only

   private:
    void _makeSamples();
    double _sampleRate;
    double _freq;
    double _amp;
    unsigned int _nSamplesInWave;
    audioSample_t *_samples;
    unsigned int _readPos;
  };

 public:
  SigGenSine(std::vector<double> &freqs, double sampleRate, double amp)
      : AudioObject("sineGen", AudioFunction::PROCESSOR), _sampleRate(sampleRate) {
    for (auto freq : freqs) {
      _sigGens.push_back(SigGenSine1Dim(freq, sampleRate, amp));
    }
  }

  ~SigGenSine() { _sigGens.clear(); }

  void step(AudioBlock &block) {
    int iChannel = 0;
    size_t blockSize = 0;
    for (auto &track : block) {
      if (iChannel == 0) {
        blockSize = track.size();
      }
      _sigGens[iChannel].step(track);
      iChannel++;
    }
    _timestamp += static_cast<mtime_t>(static_cast<double>(blockSize) / _sampleRate * 1000000);
    block.setTimestamp(_timestamp);
  }

  using AudioObject::step;

 private:
  std::vector<SigGenSine1Dim> _sigGens;
  double _sampleRate;
  mtime_t _timestamp = 0;

};  // class SigGenSine

/// Audio Processor to generate a square signal
class SigGenSquare : public AudioObject {
 public:
  SigGenSquare(double sampleRate, double period, double amp, ChannelLayout l)
      : AudioObject("sigGenSquare", AudioFunction::PROCESSOR),
        _square(l),
        _sampleRate(sampleRate),
        _periodInSamples(static_cast<int>(period * _sampleRate)),
        _lastIndex(0),
        _amp(amp) {
    // Generate one period
    for (auto &track : _square) {
      for (int i = 0; i < _periodInSamples / 2; ++i) {
        track.push_back(0);
      }
      for (int i = _periodInSamples / 2; i < _periodInSamples; ++i) {
        track.push_back(_amp);
      }
    }
  }

  ~SigGenSquare() {}

  void step(AudioBlock &block) {
    for (auto &track : block) {
      for (int i = 0; i < (int)track.size(); ++i) {
        track[(int)i] = _square[track.channel()][i % _periodInSamples + _lastIndex];
      }
    }
    _lastIndex = block.size() % _periodInSamples;
    _timestamp = static_cast<mtime_t>(static_cast<double>(block.size()) / _sampleRate * 1000000);
  }

  using AudioObject::step;

 private:
  AudioBlock _square;
  double _sampleRate;
  int _periodInSamples;
  int _lastIndex;
  double _amp;
  mtime_t _timestamp = 0;

};  // class SigGenSquare

/// Wrapper of Input::AudioReader for audiogen readers
class SigGenSineInput : public Input::AudioReader {
 public:
  SigGenSineInput(readerid_t id, SamplingRate sampleRate, ChannelLayout layout);
  SigGenSineInput(readerid_t id, SamplingRate sampleRate, ChannelLayout layout, std::vector<double> &freqs,
                  double amp = 1.);
  ~SigGenSineInput();

  Input::ReadStatus readSamples(size_t nbSamples, Audio::Samples &audioSamples);
  Status seekFrame(mtime_t date);
  size_t available();
  bool eos();

 private:
  SigGenSine _impl;
  mtime_t _timestamp;
};

}  // namespace Audio
}  // namespace VideoStitch
