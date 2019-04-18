// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audioObject.hpp"

#include <vector>

#define FILTER_BOUNDS_CHECK

namespace VideoStitch {
namespace Audio {

enum class FilterType {
  UNITY = 0,
  LOW_PASS = 1,
  HIGH_PASS = 2,
  LOW_SHELF = 5,
  HIGH_SHELF = 6,
  BELL = 7,
  DIRECT_COEFF = 8
};

// Default filter values
//
static const FilterType IIR_DEFAULT_FILTER_TYPE = FilterType::UNITY;
static const std::string IIR_NAME = "IIR filter";
static const double IIR_DEFAULT_FERQUENCY = 1000;
static const double IIR_DEFAULT_GAIN = 0;
static const double IIR_DEFAULT_Q = 1;

// IIR Filter
//
//
class IIR : public AudioObject {
 public:
  explicit IIR(int sampleRate);

  virtual ~IIR() {}

  void step(AudioBlock& out,
            const AudioBlock& in);  // Copy
  void step(AudioBlock& buf);       // In-place

  int getSampleRate() const;

  FilterType getFilterType() const;
  void setType(FilterType type);

  double getFreq() const;
  void setFreq(double freq);

  double getGain() const;
  void setGain(double gain);

  double getQ() const;
  void setQ(double q);

  void setFilterTFGQ(FilterType type, double freq, double gain, double q);

  void clearState();

 protected:
  inline void _biquad(audioSample_t& out, const audioSample_t& in);  // Copy
  inline void _biquad(audioSample_t& sample);                        // In place

 private:
  void _convertParams();
  void _convertParamsLPF();
  void _convertParamsHPF();
  void _convertParamsLS();
  void _convertParamsHS();
  void _convertParamsBell();

  int _sampleRate;

  FilterType _type;
  double _freq;
  double _gain;
  double _q;

  double _a0;
  double _a1;
  double _a2;
  double _b1;
  double _b2;

  double _d0;
  double _d1;

  audioSample_t _tmp;

};  // class AudioProcess::IIR

}  // namespace Audio
}  // namespace VideoStitch
