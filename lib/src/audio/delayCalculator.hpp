// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// A utility for calculating the time offset between two audio streams

#pragma once

#include "libvideostitch/audioBlock.hpp"

namespace VideoStitch {
namespace Audio {

/// \class DelayCalculator
/// \brief A utility for calculating the time offset between two audio streams.
///
/// Once `fill()` has been called enough times to fill the two buffers with
/// `kSearchTime` seconds of audio it will calculate the offset and return `true`.
/// Tests have shown good resistance to noise and level differences, but with very
/// low levels, time differences bigger than the search window, or sources separated
/// by very large distances.
///
/// This utility only accepts two sources, and will extract audio from the first
/// channel in each source.
///
class DelayCalculator {
 public:
  DelayCalculator();
  ~DelayCalculator();

  size_t getOffset() const;

  bool fill(const AudioBlock& early, const AudioBlock& late);
  void reset();

 private:
  std::vector<double> early_;  ///< The audio generally associated with being "real-time" (e.g. external input)
  std::vector<double> late_;   ///< The audio generally associated with being "late" (e.g. camera microphone)
  size_t numSamples_;
  size_t result_;

  void calculate_();
};

}  // end namespace Audio
}  // end namespace VideoStitch
