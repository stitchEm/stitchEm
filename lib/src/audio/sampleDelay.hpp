// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audioObject.hpp"
#include "libvideostitch/audioBlock.hpp"

#include <queue>
#include <mutex>

namespace VideoStitch {
namespace Audio {

static const double kMaxDelayTime = 20.0;  // Seconds

class SampleDelay : public AudioObject {
 public:
  static double getMaxDelaySeconds() { return kMaxDelayTime; }
  static size_t getMaxDelaySamples() { return (size_t)(kMaxDelayTime * getDefaultSamplingRate()); }

  SampleDelay();
  ~SampleDelay() {}

  Status setDelaySeconds(double delayInSeconds);
  double getDelaySeconds();
  void setDelaySamples(size_t delayInSamples);
  size_t getDelaySamples();

  void step(AudioBlock& out, const AudioBlock& in);
  void step(AudioBlock& buf);

 private:
  std::mutex delayMutex_;
  size_t curDelayTime_;

  std::map<ChannelMap, std::deque<audioSample_t>> delayBuffers_;
};

}  // namespace Audio
}  // namespace VideoStitch
