// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "stabilization/iirFilter.hpp"

namespace VideoStitch {
namespace Testing {

void testFilterDummyIIR() {
  unsigned int sampleRate = 2000;
  unsigned int fLow = 1;
  unsigned int fHigh = 500;

  unsigned int duration = 10;
  unsigned int nbSamples = sampleRate * duration;
  std::vector<double> yLow(nbSamples);       // low frequency wave
  std::vector<double> yHigh(nbSamples);      // high frequency wave
  std::vector<double> yFull(nbSamples);      // mixed low + high
  std::vector<double> yFiltered(nbSamples);  // filtered

  Stab::IIRFilter iirUninitialized;

  Stab::IIRFilter iirIdentity;
  ENSURE(iirIdentity.initLowPass(1.));

  Stab::IIRFilter iirLP;
  ENSURE(iirLP.initLowPass(100, sampleRate));

  for (std::size_t i = 0; i < nbSamples; ++i) {
    yLow[i] = std::cos(2 * M_PI * fLow * i / static_cast<double>(sampleRate));
    yHigh[i] = std::cos(2 * M_PI * fHigh * i / static_cast<double>(sampleRate));
    yFull[i] = yLow[i] + yHigh[i];
    yFiltered[i] = iirLP.filterValue(yFull[i]);

    // check identity filters
    ENSURE_APPROX_EQ(yFull[i], iirUninitialized.filterValue(yFull[i]), 1e-6);
    ENSURE_APPROX_EQ(yFull[i], iirIdentity.filterValue(yFull[i]), 1e-6);
  }

  // start at index 10 to get out of the transitional regime
  const unsigned int startIndex = 10;

  bool foundValueOverThreshold = false;
  for (std::size_t i = startIndex; i < nbSamples; ++i) {
    if (std::abs(yFull[i] - yLow[i]) > 0.5) {
      foundValueOverThreshold = true;
    }

    // make sure the filtered signal is close to the low frequency part
    ENSURE_APPROX_EQ(yLow[i], yFiltered[i], 0.07);
  }

  // make sure the full signal (high + low) is different enough from the low only
  ENSURE(foundValueOverThreshold, "No discrepancy between yFull and yLow");
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) { VideoStitch::Testing::testFilterDummyIIR(); }
