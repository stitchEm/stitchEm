// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "audioBlock.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

namespace VideoStitch {
namespace Audio {

template <typename T>
// Helper to convert a 2 dimension vector into an C-array
T** vec2array(const std::vector<std::vector<T>>& in) {
  T** out;
  size_t nchannels = in.size();
  size_t nSamples = in[0].size();
  out = (T**)malloc(nchannels * sizeof(T*));
  for (size_t c = 0; c < nchannels; c++) {
    out[c] = (T*)malloc(nSamples * sizeof(T));
    for (size_t s = 0; s < nSamples; s++) {
      out[c][s] = in[c].data()[s];
    }
  }
  return out;
}

// Helper to convert C-array into vector 2 dimensionals
template <typename T>
std::vector<std::vector<T>> array2vec(const T** in, size_t nChannels, size_t nSamples) {
  std::vector<std::vector<T>> out;
  std::vector<T> tmp;
  for (size_t c = 0; c < nChannels; c++) {
    for (size_t s = 0; s < nSamples; s++) {
      tmp.push_back(in[c][s]);
    }
    out.push_back(tmp);
  }
  return out;
}

// Helper to delete 2-dimensional audio C-array
template <typename T>
void deleteAudioArray(T** data, size_t nChannels) {
  for (size_t c = 0; c < nChannels; c++) {
    free((void*)data[c]);
  }
  free(data);
}

class VS_EXPORT AudioObject {
 public:
  enum class AudioFunction { SOURCE, PROCESSOR, SINK, UNKNOWN };

  enum class State { IDLE, BUSY, ERROR_STATE };

  AudioObject(const std::string& name, AudioFunction function)
      : _state(State::IDLE), _name(name), _function(function), _nInputs(0), _nOutputs(0), _sampleRate(0.) {}

  AudioObject(const std::string& name, AudioFunction function, unsigned int nInputs, unsigned int nOutputs,
              double sampleRate)
      : _state(State::IDLE),
        _name(name),
        _function(function),
        _nInputs(nInputs),
        _nOutputs(nOutputs),
        _sampleRate(sampleRate) {}

  virtual ~AudioObject() {}

  /**
   * @brief Process an audio block and store the result to the output audio block
   * @param out Output audio block where the result is stored
   * @param in Input audio block
   */
  virtual void step(AudioBlock& /*out*/, const AudioBlock& /*in*/) {}

  /**
   * @brief Process an audio block in place
   * @param inout Input/output audio block
   */
  virtual void step(AudioBlock& inout) = 0;

  State getState() const { return _state; }
  void setState(State state) { _state = state; }
  std::string getName() const { return _name; }
  AudioFunction getFunction() const { return _function; }

  unsigned int getInputs() const { return _nInputs; }

  void setnInputs(const unsigned int n) { _nInputs = n; }

  unsigned int getOutputs() const { return _nOutputs; }

  void setnOutputs(const unsigned int n) { _nOutputs = n; }

  double getSampleRate() const { return _sampleRate; }

  void setSampleRate(const double sr) { _sampleRate = sr; }

 protected:
  State _state;

 private:
  std::string _name;
  AudioFunction _function;
  unsigned int _nInputs;
  unsigned int _nOutputs;
  double _sampleRate;
};

}  // namespace Audio
}  // namespace VideoStitch
