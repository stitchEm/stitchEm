// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audio.hpp"

#include <string>

namespace VideoStitch {
namespace Audio {

class AudioPreProcessor {
 public:
  explicit AudioPreProcessor(const std::string& name, groupid_t gr) : name_(name), gr_(gr) {}
  virtual ~AudioPreProcessor() {}

  /** @fn void process()
   *
   * @param in  [in]  The two audio streams to be processed.
   * @param out [out] Output of the audio preprocessor.
   *
   * In must be a vector of two `Audio::Samples`. `out` must
   * be allocated to the same size as `in`.
   */
  virtual void process(const std::vector<Samples>& in, std::vector<Samples>& out) = 0;

  /** @fn void process()
   *
   * @param inOut  [in]  The two audio streams.
   * @note works in place the implementation behind should works in place also.
   *
   */
  virtual void process(std::vector<Samples>& inOut) = 0;

  /** @fn void getProcessingDelay()
   *
   * @return The induced latency in samples, per channel.
   */
  virtual float getProcessingDelay() { return 0.; }

  /** @fn void getGroup()
   *
   * @return The group id to apply this process.
   */
  groupid_t getGroup() { return gr_; }

 private:
  std::string name_;
  groupid_t gr_;
};

}  // namespace Audio
}  // namespace VideoStitch
