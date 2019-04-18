// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <audio/filter.hpp>

#include <cmath>
#include <iostream>

namespace VideoStitch {
namespace Testing {
using namespace Audio;

void testBellFilter(const int fs, const double freq, const double gain) {
  /* Generate a 1kHz sine wave */
  AudioTrack sine_1kHz(SPEAKER_FRONT_LEFT);

  for (int i = 0; i < ((fs / freq) * 1000); i++) {
    sine_1kHz.push_back(std::sin(2 * M_PI * ((double)i / ((double)fs / freq))));
  }

  /* Min/max sample value should be +/-1.0 */
  double max = 0;
  double min = 0;
  for (size_t i = 0; i < sine_1kHz.size(); i++) {
    if (sine_1kHz[i] > max) {
      max = sine_1kHz[i];
    }
    if (sine_1kHz[i] < min) {
      min = sine_1kHz[i];
    }
  }

  /* Filter the sine wave */
  AudioBlock sound(MONO, (mtime_t)0);
  sound[SPEAKER_FRONT_LEFT].swap(sine_1kHz);
  IIR filt(fs);
  filt.setFilterTFGQ(FilterType::BELL, freq, gain, 1);
  filt.step(sound);

  /* Min/max sample value should now be +/-0.5 */
  double* filtered_1kHz = reinterpret_cast<double*>(sound[SPEAKER_FRONT_LEFT].data());
  double maxFilt = 0;
  double minFilt = 0;
  /* Filter has some latency, so don't start at 0 */
  /* Delay 100 cycles */
  for (size_t i = static_cast<size_t>(100 * (fs / freq)); i < sound[SPEAKER_FRONT_LEFT].size(); i++) {
    if (filtered_1kHz[i] > maxFilt) {
      maxFilt = filtered_1kHz[i];
    }
    if (filtered_1kHz[i] < minFilt) {
      minFilt = filtered_1kHz[i];
    }
  }

  /* No values should be 0 */
  ENSURE(min != 0);
  ENSURE(max != 0);
  ENSURE(minFilt != 0);
  ENSURE(maxFilt != 0);

  /* Convert dB gain applied to a linear value and compare to the gain applied */
  double ratio = std::pow(10, gain / 20.0);
  ENSURE_APPROX_EQ(min * ratio, minFilt, 0.0000001);
  ENSURE_APPROX_EQ(max * ratio, maxFilt, 0.0000001);
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int /* argc */, char** /* argv */) {
  VideoStitch::Testing::initTest();

  /*
   * The Bell filter uses all parameters (frequency, gain, and Q)
   */

  /* **** 44.1 kHz audio **** */

  /* Filter 100Hz, -1dB */
  VideoStitch::Testing::testBellFilter(44100, 100.0, -1.0);
  /* Filter 1kHz, -6.02dB (~0.5) */
  VideoStitch::Testing::testBellFilter(44100, 1000.0, -6.02);
  /* Filter 10kHz, -20dB (0.1) */
  VideoStitch::Testing::testBellFilter(44100, 10000.0, -20.0);

  /* **** 48 kHz audio **** */

  /* Filter 100Hz, -1dB */
  VideoStitch::Testing::testBellFilter(48000, 100.0, -1.0);
  /* Filter 1kHz, -6.02dB (~0.5) */
  VideoStitch::Testing::testBellFilter(48000, 1000.0, -6.02);
  /* Filter 10kHz, -20dB (0.1) */
  VideoStitch::Testing::testBellFilter(48000, 10000.0, -20.0);

  return 0;
}
