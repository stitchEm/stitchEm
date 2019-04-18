// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <cmath>
#include <vector>
#include <cassert>
#include <memory>
#include <mutex>
#include "libvideostitch/audio.hpp"
#include "libvideostitch/circularBuffer.hpp"
#include "libvideostitch/audioObject.hpp"

namespace VideoStitch {
namespace Audio {

static const size_t defaultSmoothing = 1000;

class Smoother {
 public:
  explicit Smoother(size_t size) : data(size), accu(0), smoothing_(size) {}

  void push(audioSample_t x) {
    accu += x * x;
    data.push(x);
    if (data.size() > smoothing_) {
      accu -= data[0] * data[0];
      data.erase(1);
    }
  }

  double getRmsValue() { return std::sqrt(accu / static_cast<double>(smoothing_)); }

  void setSmoothing(size_t smoothing) { smoothing_ = smoothing; }

 private:
  CircularBuffer<audioSample_t> data;
  double accu;
  size_t smoothing_;
};

enum class DetectorType { PEAK, RMS };

class EnvelopeDetector {
 public:
  EnvelopeDetector(const double fs, const DetectorType type = DetectorType::PEAK, const double attack = 0.005,
                   const double hold = 0.000, const double release = 1.400);
  ~EnvelopeDetector();

  void setAttack(double attack) {
    assert(attack >= 1.0e-6 && attack <= 5.0);  // Attack time out of range
    attackTime_ = attack;
    attackGain_ = std::exp(-1.0 / (fs_ * attackTime_));
  }
  double getAttack() const { return attackTime_; }
  void setRelease(double release) {
    assert(release >= 1.0e-6 && release <= 5.0);  // Release time out of range
    releaseTime_ = release;
    releaseGain_ = std::exp(-1.0 / (fs_ * releaseTime_));
  }
  double getRelease() const { return releaseTime_; }
  void setHold(double hold) {
    assert(hold >= 1.0e-6 && hold <= 5.0);  // Hold time out of range
    holdTime_ = hold;
  }
  double getHold() { return holdTime_; }

  void setSmoothing(size_t smoothing) {
    assert(type_ == DetectorType::RMS);  // Wrong detector type
    assert(smoothing < 1000000);         // Smoothing value out of range
    smoother_.setSmoothing(smoothing);
  }

  double getEnvelope() const;
  void process(const AudioTrack& in, AudioTrack* out = nullptr);

 private:
  double fs_;
  DetectorType type_;
  Smoother smoother_;

  double attackTime_;
  double holdTime_;
  double releaseTime_;

  double attackGain_;
  double releaseGain_;

  int holdInSamples_;
  int holdCount_;
  double envelopeSample_;
};

class VuMeter : public AudioObject {
 public:
  explicit VuMeter(int sr);
  ~VuMeter() {}

  double getPeakAttack() const;
  std::vector<double> getPeakValues() const;
  double getRmsAttack() const;
  std::vector<double> getRmsValues() const;
  const AudioBlock& getPeakEnvelope() const;
  double getPeakRelease() const;
  double getRmsRelease() const;
  const AudioBlock& getRmsEnvelope() const;
  void setSmoothing(size_t smoothing);
  void setDebug(bool b);
  void setPeakAttack(double attack);
  void setPeakRelease(double release);
  void setRmsAttack(double attack);
  void setRmsRelease(double release);

  void step(AudioBlock& buf);

 private:
  bool initialized_;
  int samplingRate_;
  std::vector<EnvelopeDetector> peakDetectors_;
  std::vector<EnvelopeDetector> rmsDetectors_;
  AudioBlock debugPeakBlk_;
  AudioBlock debugRmsBlk_;
  bool debug_ = false;
  double peakAttack_;
  double peakRelease_;
  double rmsAttack_;
  double rmsRelease_;
  mutable std::mutex paramsLock_;
};

}  // namespace Audio
}  // namespace VideoStitch
