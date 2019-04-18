// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "envelopeDetector.hpp"
#include "gain.hpp"

namespace VideoStitch {
namespace Audio {

EnvelopeDetector::EnvelopeDetector(const double fs, const DetectorType type, const double attack, const double hold,
                                   const double release)
    : fs_(fs),
      type_(type),
      smoother_((type == DetectorType::RMS) ? defaultSmoothing : 0)  // use a smoothing window for the RMS detector only
      ,
      attackTime_(attack),
      holdTime_(hold),
      releaseTime_(release),
      holdCount_(0),
      envelopeSample_(0) {
  assert(fs_ >= 32000 && fs_ <= 384000);                  // Sample rate out of range
  assert(attackTime_ >= 1.0e-6 && attackTime_ <= 5.0);    // Attack time out of range
  assert(releaseTime_ >= 1.0e-6 && releaseTime_ <= 5.0);  // Release time out of range
  assert(holdTime_ >= 0 && holdTime_ <= 1.0);             // Hold time out of range

  attackGain_ = std::exp(-1.0 / (fs_ * attackTime_));
  releaseGain_ = std::exp(-1.0 / (fs_ * releaseTime_));
  holdInSamples_ = static_cast<int>(fs_ * holdTime_);
}

EnvelopeDetector::~EnvelopeDetector() {}

double EnvelopeDetector::getEnvelope() const { return envelopeSample_; }

void EnvelopeDetector::process(const AudioTrack &in, AudioTrack *out) {
  for (auto x : in) {
    double inputSample;
    if (type_ == DetectorType::RMS) {
      smoother_.push(x);
      inputSample = smoother_.getRmsValue();
    } else {
      inputSample = std::abs(x);
    }
    if (inputSample > envelopeSample_) {
      holdCount_ = 0;
      envelopeSample_ = inputSample + (attackGain_ * (envelopeSample_ - inputSample));
    } else {
      if (holdCount_ < holdInSamples_) {
        holdCount_++;
      } else {
        envelopeSample_ = inputSample + (releaseGain_ * (envelopeSample_ - inputSample));
      }
    }
    if (out) {
      out->push_back(envelopeSample_);
    }
  }
}

VuMeter::VuMeter(int sr)
    : AudioObject("vumeter", AudioFunction::SINK),
      initialized_(false),
      samplingRate_(sr),
      peakAttack_(0.005),
      peakRelease_(1.4),
      rmsAttack_(0.1),
      rmsRelease_(0.3) {}

double VuMeter::getPeakAttack() const {
  assert(!peakDetectors_.empty());
  return peakDetectors_.begin()->getAttack();
}

std::vector<double> VuMeter::getPeakValues() const {
  std::vector<double> peaks;
  for (auto &peakDetector : peakDetectors_) {
    peaks.push_back(linToDB(peakDetector.getEnvelope()));
  }
  return peaks;
}

double VuMeter::getRmsAttack() const {
  assert(!rmsDetectors_.empty());
  return rmsDetectors_.begin()->getAttack();
}

std::vector<double> VuMeter::getRmsValues() const {
  std::vector<double> rms;
  for (auto &detector : rmsDetectors_) {
    rms.push_back(linToDB(detector.getEnvelope()));
  }
  return rms;
}

const AudioBlock &VuMeter::getPeakEnvelope() const {
  assert(debug_);
  return debugPeakBlk_;
}

double VuMeter::getPeakRelease() const {
  assert(!peakDetectors_.empty());
  return peakDetectors_.begin()->getRelease();
}

double VuMeter::getRmsRelease() const {
  assert(!rmsDetectors_.empty());
  return rmsDetectors_.begin()->getRelease();
}

const AudioBlock &VuMeter::getRmsEnvelope() const {
  assert(debug_);
  return debugRmsBlk_;
}

void VuMeter::setSmoothing(size_t smoothing) {
  for (size_t i = 0; i < rmsDetectors_.size(); ++i) {
    rmsDetectors_.at(i).setSmoothing(smoothing);
  }
}

void VuMeter::setDebug(bool b) { debug_ = b; }

void VuMeter::setPeakAttack(double attack) {
  std::lock_guard<std::mutex> lk(paramsLock_);
  peakAttack_ = attack;
  for (EnvelopeDetector &detector : peakDetectors_) {
    detector.setAttack(attack);
  }
}

void VuMeter::setPeakRelease(double release) {
  std::lock_guard<std::mutex> lk(paramsLock_);
  peakRelease_ = release;
  for (EnvelopeDetector &detector : peakDetectors_) {
    detector.setRelease(release);
  }
}

void VuMeter::setRmsAttack(double attack) {
  std::lock_guard<std::mutex> lk(paramsLock_);
  rmsAttack_ = attack;
  for (EnvelopeDetector &detector : rmsDetectors_) {
    detector.setAttack(attack);
  }
}

void VuMeter::setRmsRelease(double release) {
  std::lock_guard<std::mutex> lk(paramsLock_);
  rmsRelease_ = release;
  for (EnvelopeDetector &detector : rmsDetectors_) {
    detector.setRelease(release);
  }
}

void VuMeter::step(AudioBlock &buf) {
  std::lock_guard<std::mutex> lk(paramsLock_);
  if (!initialized_) {
    int nbChannels = getNbChannelsFromChannelLayout(buf.getLayout());
    Logger::get(Logger::Verbose) << "Initialize vumeter with nb channels " << nbChannels << " layout "
                                 << getStringFromChannelLayout(buf.getLayout()) << std::endl;
    for (int i = 0; i < nbChannels; i++) {
      peakDetectors_.emplace_back(samplingRate_, DetectorType::PEAK, peakAttack_, 0., peakRelease_);
      rmsDetectors_.emplace_back(samplingRate_, DetectorType::RMS, rmsAttack_, 0., rmsRelease_);
    }
    initialized_ = true;
  }

  if (debug_) {
    if (debugPeakBlk_.empty()) {
      debugPeakBlk_.setChannelLayout(buf.getLayout());
      debugPeakBlk_.setTimestamp(buf.getTimestamp());
      debugRmsBlk_.setChannelLayout(buf.getLayout());
      debugRmsBlk_.setTimestamp(buf.getTimestamp());
    }

    int i = 0;
    for (const AudioTrack &track : buf) {
      peakDetectors_.at(i).process(track, &debugPeakBlk_[track.channel()]);
      rmsDetectors_.at(i).process(track, &debugRmsBlk_[track.channel()]);
      i++;
    }
  } else {
    int i = 0;
    for (const AudioTrack &track : buf) {
      peakDetectors_.at(i).process(track);
      rmsDetectors_.at(i).process(track);
      i++;
    }
  }
}

}  // namespace Audio
}  // namespace VideoStitch
