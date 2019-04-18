// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gain.hpp"
#include "math.h"

namespace VideoStitch {
namespace Audio {

gainDB_t clampGain(gainDB_t x) {
  if (x < kGainMin) {
    Logger::get(Logger::Warning) << "Gain " << x << " < than min value accepted " << kGainMin;
    return kGainMin;
  } else if (x > kGainMax) {
    Logger::get(Logger::Warning) << "Gain " << x << " > than max value accepted " << kGainMax;
    return kGainMax;
  }
  return x;
}

gainLin_t dBToLin(gainDB_t x) { return pow(10., x / 20.); }

gainPercent_t dBToPercent(gainDB_t x) { return (x - kGainMin) * 100. / (kGainMax - kGainMin); }

gainDB_t linToDB(gainLin_t x) { return 20. * log10(x); }

gainPercent_t linToPercent(gainLin_t x) { return dBToPercent(linToDB(x)); }

gainDB_t percentToDB(gainPercent_t x) { return x * (kGainMax - kGainMin) / 100. + kGainMin; }

gainLin_t percentToLin(gainPercent_t x) { return (dBToLin(percentToDB(x))); }

Gain::Gain(gainDB_t gaindB, bool reversePolarity, bool mute)
    : AudioObject("gain", AudioFunction::PROCESSOR),
      gain_(dBToLin(gaindB)),
      reversePolarity_(reversePolarity),
      mute_(mute) {}

gainDB_t Gain::getGainDB() { return linToDB(gain_); }

gainPercent_t Gain::getGainPercent() { return linToPercent(gain_); }

bool Gain::getMute() { return mute_; }

bool Gain::getReversePolarity() { return reversePolarity_; }

void Gain::setGainDB(gainDB_t gain) { gain_ = dBToLin(clampGain(gain)); }

void Gain::setGainPercent(gainPercent_t gain) { gain_ = dBToLin(clampGain(percentToDB(gain))); }

void Gain::setMute(bool m) { mute_ = m; }

void Gain::setReversePolarity(bool r) { reversePolarity_ = r; }

void Gain::step(AudioBlock &out, const AudioBlock &in) {
  AudioBlock workingBlock = in.clone();
  step(workingBlock);
  out = std::move(workingBlock);
}

void Gain::step(AudioBlock &inout) {
  if (mute_) {
    for (auto &track : inout) {
      memset(track.data(), 0, track.size() * sizeof(audioSample_t));
    }
  } else {
    gainLin_t gain;

    if (reversePolarity_) {
      gain = -gain_;
    } else {
      gain = gain_;
    }

    for (auto &track : inout) {
      for (auto &s : track) {
        s *= gain;
      }
    }
  }
}

}  // namespace Audio
}  // namespace VideoStitch
