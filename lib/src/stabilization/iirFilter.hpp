// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

namespace VideoStitch {
namespace Stab {

class IIRFilter {
 public:
  IIRFilter();

  /**
   * @brief init biquad parameters
   * @param cutoffFreq : cutoff frequency
   * @param sampleRate : sampling rate
   */
  Status initLowPass(unsigned int cutoffFreq, unsigned int sampleRate);

  /**
   * @brief init biquad parameters given a ratio of Nyquist frequency
   * @param ratioNyquist : between 0 and 1
   */
  Status initLowPass(double ratioNyquist);

  /**
   * @brief Compute the next filtered value
   * @param inputVal : input value
   * @return output value
   *
   * This function uses the state (x1, x2, y1, y2) of the object to
   * compute the next value. It also updates this state during the process
   *
   * see: http://www.earlevel.com/main/2003/02/28/biquads/
   *
   */
  double filterValue(double inputVal);

 private:
  void initLowPassHelper(double K);

  double a0, a1, a2, b1, b2;
  double x1;  /// order 1 delayed input
  double x2;  /// order 2 delayed input
  double y1;  /// order 1 delayed output
  double y2;  /// order 2 delayed output
};

}  // namespace Stab
}  // namespace VideoStitch
