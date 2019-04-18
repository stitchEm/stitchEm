// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "iirFilter.hpp"
#include "libvideostitch/logging.hpp"

#include <cmath>

namespace VideoStitch {
namespace Stab {

IIRFilter::IIRFilter() : a0(1.), a1(0.), a2(0.), b1(0.), b2(0.), x1(0.), x2(0.), y1(0.), y2(0.) {}

Status IIRFilter::initLowPass(unsigned int freq, unsigned int sampleRate) {
  if ((freq == 0) || (sampleRate == 0) || (static_cast<double>(freq) > sampleRate / 2.)) {
    std::string errMsg = "IIRFilter::initLowPass(): bad combinaison of freq and sampleRate";
    Logger::get(Logger::Error) << errMsg << std::endl;
    return Status(Origin::StabilizationAlgorithm, ErrType::InvalidConfiguration, errMsg);
  }
  double K = std::tan(M_PI * freq * (1.0 / sampleRate));
  initLowPassHelper(K);
  return Status::OK();
}

Status IIRFilter::initLowPass(double ratioNyquist) {
  if ((ratioNyquist <= 0) || (ratioNyquist > 1)) {
    std::string errMsg = "IIRFilter::initLowPass(): bad ratioNyquist value";
    Logger::get(Logger::Error) << errMsg << std::endl;
    return Status(Origin::StabilizationAlgorithm, ErrType::InvalidConfiguration, errMsg);
  }
  double K = std::tan(ratioNyquist * M_PI / 2.);
  initLowPassHelper(K);
  return Status::OK();
}

void IIRFilter::initLowPassHelper(double K) {
  double K2 = K * K;

  double n = 1.0 / (1.0 + M_SQRT2 * K + K2);
  a0 = K2 * n;
  a1 = 2. * a0;
  a2 = a0;
  b1 = 2. * (K2 - 1.) * n;
  b2 = (1. - M_SQRT2 * K + K2) * n;
  x1 = x2 = y1 = y2 = 0.;
}

double IIRFilter::filterValue(double inputVal) {
  double outputVal = inputVal * a0 + x1 * a1 + x2 * a2 - y1 * b1 - y2 * b2;
  x2 = x1;
  x1 = inputVal;
  y2 = y1;
  y1 = outputVal;
  return outputVal;
}

}  // namespace Stab
}  // namespace VideoStitch
