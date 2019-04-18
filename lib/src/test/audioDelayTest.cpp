// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "audio/sampleDelay.hpp"
#include "libvideostitch/audioWav.hpp"

#include <cmath>
#include <iostream>

namespace VideoStitch {

using namespace Audio;

namespace Testing {

void testDelay() {
  initTest();

  SampleDelay* del = new SampleDelay();

  // Test setters and getters that convert time to samples
  del->setDelaySeconds(1);
  ENSURE(del->getDelaySamples() == getDefaultSamplingRate(), "setGlobalDelaySeconds(1) getDelaySamples");
  ENSURE(del->getDelaySeconds() == 1.0, "setGlobalDelaySeconds(1) getDelaySeconds");
  del->setDelaySeconds(0);
  ENSURE(del->getDelaySamples() == 0, "setGlobalDelaySeconds(0) getDelaySamples");
  ENSURE(del->getDelaySeconds() == 0, "setGlobalDelaySeconds(0) getDelaySeconds");

  del->setDelaySeconds(0.5);
  ENSURE(del->getDelaySamples() == getDefaultSamplingRate() / 2, "setDelaySeconds(i, 0.5) getDelaySamples(i)");
  ENSURE(del->getDelaySeconds() == 0.5, "setDelaySeconds(i, 0.5) getDelaySeconds(i)");

  delete del;

  del = new SampleDelay();
  AudioBlock a(Audio::MONO);
  a[SPEAKER_FRONT_LEFT].assign(9, 0.0);
  a[SPEAKER_FRONT_LEFT].push_back(1.0);
  del->setDelaySamples(5);

  // The first time we call step, there should not be any delay
  del->step(a);  // Test in place
  //  in: 0 0 0 0 0 0 0 0 0 1
  // out: 0 0 0 0 0 0 0 0 0 1
  ENSURE(a[SPEAKER_FRONT_LEFT].size() == 10, "First time process - size");
  ENSURE(a[SPEAKER_FRONT_LEFT][9] == 1.0, "First time process - value");

  // Second time we have enough samples to delay. The applied delay
  // will change so a window will be applied. The delayed output sample
  // will be between 0 and 1.
  AudioBlock out(MONO);
  del->step(a, out);  // Test copy
  // delay: 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1
  //   out:           0 0 0 0 X 0 0 0 0 0
  ENSURE(out[SPEAKER_FRONT_LEFT].size() == 10, "Second time process - size");
  std::cout << out[SPEAKER_FRONT_LEFT][4] << std::endl;
  ENSURE(out[SPEAKER_FRONT_LEFT][4] != 0.0, "Second time process - value");  // Between 0...
  ENSURE(out[SPEAKER_FRONT_LEFT][4] != 1.0, "Second time process - value");  // ...and 1

  // Third time, the delay will not change, so we will not window
  //  and the output sample should not be attenuated.
  del->step(a, out);
  // delay: 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1
  //   out:                               0 0 0 0 1 0 0 0 0 0
  ENSURE(out[SPEAKER_FRONT_LEFT].size() == 10, "Third time process - size");
  ENSURE(out[SPEAKER_FRONT_LEFT][4] == 1.0, "Third time process - value");

  delete del;
}

void testSinusDelay() {
  std::string inputFile = "/tmp/inputDelay.wav";
  std::string outputFile = "/tmp/outputDelay.wav";
  WavReader wr(inputFile.c_str());
  WavWriter ww(outputFile, ChannelLayout::MONO, getDefaultSamplingRate());
  uint32_t blockSize = 470;
  size_t nSamples = static_cast<size_t>(wr.getnSamples());
  size_t nBlock = nSamples / blockSize;
  AudioBlock inout;

  SampleDelay* del = new SampleDelay();
  del->setDelaySamples(44100);

  for (size_t i = 0; i < nBlock; i++) {
    wr.step(inout, blockSize);
    ENSURE_EQ((size_t)blockSize, inout.begin()->size(), "unexpected read samples");
    del->step(inout);
    ww.step(inout);
    inout.clear();
  }
  ww.close();
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /*argc*/, char** /*argv*/) {
  //  VideoStitch::Testing::testDelay();
  //  VideoStitch::Testing::testSinusDelay();
  return 0;
}
