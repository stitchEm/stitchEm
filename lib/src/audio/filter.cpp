// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "filter.hpp"

#include <cmath>
#include <cassert>

namespace VideoStitch {
namespace Audio {

IIR::IIR(int sampleRate)
    : AudioObject(IIR_NAME, AudioFunction::PROCESSOR, 1, 1, (double)sampleRate),
      _type(IIR_DEFAULT_FILTER_TYPE),
      _freq(IIR_DEFAULT_FERQUENCY),
      _gain(IIR_DEFAULT_GAIN),
      _q(IIR_DEFAULT_Q),
      _d0(0),
      _d1(0),
      _tmp(0) {
  assert(sampleRate >= 32000);
  _sampleRate = sampleRate;
  _convertParams();
}

int IIR::getSampleRate() const { return _sampleRate; }

FilterType IIR::getFilterType() const { return _type; }
void IIR::setType(FilterType type) {
  _type = type;
  _convertParams();
}

double IIR::getFreq() const { return _freq; }
void IIR::setFreq(double freq) {
  assert(freq > 0);
  _freq = freq;
  _convertParams();
}

double IIR::getGain() const { return _gain; }
void IIR::setGain(double gain) {
  _gain = gain;
  _convertParams();
}

double IIR::getQ() const { return _q; }

void IIR::setQ(double q) {
  assert(q > 0);
  _q = q;
  _convertParams();
}

void IIR::setFilterTFGQ(FilterType type, double freq, double gain, double q) {
  assert(freq > 0);
  assert(q > 0);
  _type = type;
  _freq = freq;
  _gain = gain;
  _q = q;
  clearState();
  _convertParams();
}

void IIR::clearState() {
  _d0 = 0;
  _d1 = 0;
}

// Convert human-readable filter parameters into biquad coefficients
//

void IIR::_convertParams() {
  switch (_type) {
    case FilterType::UNITY:
      _a0 = 1;
      _a1 = _a2 = _b1 = _b2 = 0;
      break;
    case FilterType::LOW_PASS:
      _convertParamsLPF();
      break;
    case FilterType::HIGH_PASS:
      _convertParamsHPF();
      break;
    case FilterType::LOW_SHELF:
      _convertParamsLS();
      break;
    case FilterType::HIGH_SHELF:
      _convertParamsHS();
      break;
    case FilterType::BELL:
      _convertParamsBell();
      break;
    case FilterType::DIRECT_COEFF:
      break;
    default:
      assert(false);
  }
}

#define A (std::pow(10.0, std::abs(_gain) * 0.05))
#define K (std::tan(M_PI * _freq * (1.0 / _sampleRate)))
#define K2 (K * K)
#define AK2 (A * K2)
#define Sq2_K (M_SQRT2 * K)
#define Sq2A_K (std::sqrt(2.0 * A) * K)

void IIR::_convertParamsLPF() {
  double n = 1.0 / (1.0 + Sq2_K + K2);

  _a0 = K2 * n;
  _a1 = 2.0 * _a0;
  _a2 = _a0;
  _b1 = 2.0 * (K2 - 1.0) * n;
  _b2 = (1.0 - Sq2_K + K2) * n;
}

void IIR::_convertParamsHPF() {
  double n = 1.0 / (1.0 + Sq2_K + K2);

  _a0 = n;
  _a1 = -2.0 * n;
  _a2 = n;
  _b1 = 2.0 * (K2 - 1.0) * n;
  _b2 = (1.0 - Sq2_K + K2) * n;
}

void IIR::_convertParamsLS() {
  double n;

  if (_gain >= 0) {
    n = 1.0 / (1.0 + Sq2_K + K2);
    _a0 = (1.0 + Sq2A_K + AK2) * n;
    _a1 = (2.0 * (AK2 - 1.0)) * n;
    _a2 = (1.0 - Sq2A_K + AK2) * n;
    _b1 = (2.0 * (K2 - 1.0)) * n;
    _b2 = (1.0 - Sq2_K + K2) * n;
  } else {
    n = 1.0 / (1.0 + Sq2A_K + AK2);
    _a0 = (1.0 + Sq2_K + K2) * n;
    _a1 = (2.0 * (K2 - 1.0)) * n;
    _a2 = (1.0 - Sq2_K + K2) * n;
    _b1 = (2.0 * (AK2 - 1.0)) * n;
    _b2 = (1.0 - Sq2A_K + AK2) * n;
  }
}

void IIR::_convertParamsHS() {
  double n;

  if (_gain >= 0) {
    n = 1.0 / (1.0 + Sq2_K + K2);
    _a0 = (A + Sq2A_K + K2) * n;
    _a1 = (2.0 * (K2 - A)) * n;
    _a2 = (A - Sq2A_K + K2) * n;
    _b1 = (2.0 * (K2 - 1.0)) * n;
    _b2 = (1.0 - Sq2_K + K2) * n;
  } else {
    n = 1.0 / (A + Sq2A_K + K2);
    _a0 = (1 + Sq2_K + K2) * n;
    _a1 = (2.0 * (K2 - 1.0)) * n;
    _a2 = (1 - Sq2_K + K2) * n;
    _b1 = (2.0 * (K2 - A)) * n;
    _b2 = (A - Sq2A_K + K2) * n;
  }
}

void IIR::_convertParamsBell() {
  double KoQ = K / _q;
  double AKoQ = A * KoQ;

  double n;

  if (_gain >= 0) {
    n = 1.0 / (1.0 + KoQ + K2);
    _a0 = (1.0 + AKoQ + K2) * n;
    _a1 = (2.0 * (K2 - 1.0)) * n;
    _a2 = (1.0 - AKoQ + K2) * n;
    _b1 = _a1;
    _b2 = (1.0 - KoQ + K2) * n;
  } else {
    n = 1.0 / (1.0 + AKoQ + K2);
    _a0 = (1 + KoQ + K2) * n;
    _a1 = (2.0 * (K2 - 1.0)) * n;
    _a2 = (1.0 - KoQ + K2) * n;
    _b1 = _a1;
    _b2 = (1.0 - AKoQ + K2) * n;
  }
}

#undef A
#undef K
#undef K2
#undef AK2
#undef Sq2_K
#undef Sq2A_K

// Direct Form 2 Transposed biquads
//
// It is generally accepted that DF2T is the best
// form for floating point calculations.
//
// See http://www.earlevel.com/main/2003/02/28/biquads/
//
inline void IIR::_biquad(audioSample_t& out, const audioSample_t& in) {
  out = (audioSample_t)((_a0 * (double)in) + _d0);

  _d0 = (_a1 * (double)(in)) + (-1 * _b1 * (double)out) + _d1;
  _d1 = (_a2 * (double)(in)) + (-1 * _b2 * (double)out);
}

inline void IIR::_biquad(audioSample_t& sample) {
  _tmp = sample;

  sample = (audioSample_t)((_a0 * (double)_tmp) + _d0);

  _d0 = (_a1 * (double)(_tmp)) + (-1 * _b1 * (double)sample) + _d1;
  _d1 = (_a2 * (double)(_tmp)) + (-1 * _b2 * (double)sample);
}

// Copy
void IIR::step(AudioBlock& out, const AudioBlock& in) {
#if defined(FILTER_BOUNDS_CHECK)
  assert(out[0].size() == in[0].size());
  for (channel_t i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    assert(out[i].size() == in[i].size());
  }
#endif

  this->setState(State::BUSY);

  size_t n = (size_t)out[0].size();

  uint32_t mask = 0x1;
  for (channel_t i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (mask & out.getLayout()) {
      for (size_t j = 0; j < n; j++) {
        _biquad(out[i][j], in[i][j]);
      }
    }
    mask <<= 1;
  }

  this->setState(State::IDLE);
}

// In-place
void IIR::step(AudioBlock& buf) {
  this->setState(State::BUSY);

  for (auto& track : buf) {
    for (auto& sample : track) {
      _biquad(sample);
    }
  }

  this->setState(State::IDLE);
}

}  // namespace Audio
}  // namespace VideoStitch
