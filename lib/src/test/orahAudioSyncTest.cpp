// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#if !defined(__clang_analyzer__)

#include "audio/orah/orahAudioSync.hpp"

#include "gpu/testing.hpp"
#include "libvideostitch/logging.hpp"

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <utility>
#include <thread>

// Uncomment to process audio and dump to raw PCM file then return.
// To be used for listening tests. Set output filename as well.
//
//#define DUMP_FILE   "/users/videostitch/test.dat"

#define _LOG Logger::get(Logger::Info)

namespace VideoStitch {
namespace Testing {

typedef Audio::Orah::orahSample_t sample_t;

static const int blockSize{1024};  // 512 samples * 2 (interleaved) channels
#ifdef DUMP_FILE
static const int realDataSize{(int)(44100 * 2 * 20)};  // 20 seconds
#else
static const int realDataSize{(int)(44100 * 2 * 7)};  // 7 seconds, more than kSyncTimeout
#endif
static const int fakeDataSize{blockSize * 20};  // 20 blocks
static const int kClickDelay{110};              // See orahAudioSync.cpp

// Create blanks.
// Mark one sample before and one after to check for correct blanking
static void genData(sample_t *buf, int len, int offset) {
  memset(buf, 4, sizeof(sample_t) * len);
  for (int i = 0, k = 0; i < ORAH_SYNC_NUM_BLANKS; i++) {
    if (offset > 0 || i != 0) {
      buf[i * 88 * 2 + k - 1 + offset] = 12345;
    }
    for (int j = 0; j < 88; j++, k++) {
      buf[i * 88 * 2 + j + offset] = 0;
    }
    buf[i * 88 * 2 + k + offset] = 12345;
  }
}

static std::ifstream f;
static void readData(sample_t *buf1, sample_t *buf2, int len) {
  sample_t s[4];
  for (int i = 0; i < len;) {
    f.read((char *)&s[0], 8);
    buf1[i] = s[0];
    buf2[i++] = s[2];
    buf1[i] = s[1];
    buf2[i++] = s[3];
  }
}

#ifdef DUMP_FILE
static std::ofstream ofile;
static void writeOut(std::vector<Audio::Samples> &out) {
  auto *s1 = ((sample_t **)out[0].getSamples())[0];
  auto *s2 = ((sample_t **)out[1].getSamples())[0];

  for (int i = 0; i < blockSize; i += 2) {
    ofile.write((char *)&s1[i], 4);
    ofile.write((char *)&s2[i], 4);
  }
}
#endif

static int testAudioSync(const char *fname) {
  f.open(fname, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Could not open file " << fname << " for reading." << std::endl;
    return -1;
  }
  f.seekg(0x2c, std::ios::beg);  // Start of audio data

#ifdef DUMP_FILE
  ofile.open(DUMP_FILE, std::ios::out | std::ios::binary);
  if (!ofile.is_open()) {
    std::cerr << "Could not open file for writing." << std::endl;
    return -1;
  }
#endif

  Audio::Orah::OrahAudioSync oas0(Audio::BlockSize::BS_512);  // Must be blockSize / 2
  ENSURE(0 == oas0.getOffset(), "Initial offset value should be 0.");

  // Buffers
  //
  // Output
  std::vector<Audio::Samples> out;
  Audio::Samples::data_buffer_t blockOut1;
  Audio::Samples::data_buffer_t blockOut2;
  blockOut1[0] = new uint8_t[blockSize * sizeof(sample_t)];
  blockOut2[0] = new uint8_t[blockSize * sizeof(sample_t)];
  Audio::Samples o1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                    blockOut1, blockSize);
  Audio::Samples o2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                    blockOut2, blockSize);
  out.push_back(std::move(o1));
  out.push_back(std::move(o2));
  // Generated data
  std::vector<sample_t> data[2];  // Two-channel "streams"
  data[0].resize(fakeDataSize);
  data[1].resize(fakeDataSize);
  // Input
  Audio::Samples::data_buffer_t block1;
  Audio::Samples::data_buffer_t block2;

  // Test timeout/cross-correlation on real data
  _LOG << std::endl << "==== Testing real data: Cross-correlation 0 ====" << std::endl;
  int sCount = 0;
  int n = 0;
  while (n < realDataSize - blockSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    readData((sample_t *)block1[0], (sample_t *)block2[0], blockSize);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s2));
    in.push_back(std::move(s1));
    oas0.process(in, out);
#ifdef DUMP_FILE
    writeOut(out);
    n += blockSize;
  }
  ofile.close();
  return 0;
#else
    n += blockSize;
    sCount += blockSize;
  }
#endif
  _LOG << "====" << std::endl;
  float os = oas0.getOffsetBlocking();  // Will wait for cross-correlation to finish if running
  _LOG << "Offset: " << os << std::endl;
  ENSURE(3088 == os, "Xcorr offset calculation failed");
  _LOG << std::endl;

  // Rewind audio file
  f.seekg(0x2c, std::ios::beg);                              // Start of audio data
  Audio::Orah::OrahAudioSync oas(Audio::BlockSize::BS_512);  // Must be blockSize / 2
  // Test timeout/cross-correlation on real data
  _LOG << "==== Testing real data: Cross-correlation 1 ====" << std::endl;
  sCount = 0;
  n = 0;
  while (n < realDataSize - blockSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    readData((sample_t *)block1[0], (sample_t *)block2[0], blockSize);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    n += blockSize;
    sCount += blockSize;
  }
  _LOG << "====" << std::endl;
  os = oas.getOffsetBlocking();  // Will wait for cross-correlation to finish if running
  _LOG << "Offset: " << os << std::endl;
  ENSURE(3088 == os, "Xcorr offset calculation failed");
  _LOG << std::endl;

  _LOG << "==== Testing real data: Blanks and masking ====" << std::endl;
  // Keep looking for blanks (there are some!) and when the
  // delay value changes to 3086, we stop and check that
  // the blanks have been correctly erased.
  n = 0;
  while (n < realDataSize - blockSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    readData((sample_t *)block1[0], (sample_t *)block2[0], blockSize);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    if (oas.getOffset() != os) {
      break;
    }
    n += blockSize;
    sCount += blockSize;
  }
  _LOG << "====" << std::endl;
  _LOG << "Offset: " << oas.getOffset() << std::endl;
  ENSURE(oas.getOffset() == 3086, "Bad detected offset");
  // We don't really care what the blanks were replaced with, as
  // this is tuned through listening tests, just that there is no
  // more string of zeros.
  std::vector<sample_t> a1(&((sample_t **)out[0].getSamples().data())[0][0],
                           &((sample_t **)out[0].getSamples().data())[0][blockSize]);
  std::vector<sample_t> a2(&((sample_t **)out[1].getSamples().data())[0][0],
                           &((sample_t **)out[1].getSamples().data())[0][blockSize]);
  auto sk1 = std::search_n(a1.begin(), a1.end(), 15, 0);
  auto sk2 = std::search_n(a2.begin(), a2.end(), 15, 0);
  if (sk1 != a1.end()) {
    auto pos = std::distance(a1.begin(), sk1);
    _LOG << "Suspicious zeros at " << pos << " of current block. (" << pos + sCount << ")" << std::endl;
  }
  if (sk2 != a2.end()) {
    auto pos = std::distance(a2.begin(), sk2);
    _LOG << "Suspicious zeros at " << pos << " of current block. (" << pos + sCount << ")" << std::endl;
  }
  ENSURE(sk1 == a1.end(), "Blank 1 not erased");
  ENSURE(sk2 == a2.end(), "Blank 2 not erased");
  _LOG << std::endl;

  // Turn off click suppression for the next few tests so we can
  // detect that the blanks are correctly aligned, and that samples
  // are correctly interpolated.
  oas.diableClickSuppresion(true);

  int bPos;

#if 1
  _LOG << "==== Testing fake data: Blanks, half sample offset ====" << std::endl;
  // Test half-sample detection on synthetic data
  bPos = 151;
  genData(data[0].data(), fakeDataSize, 0);
  genData(data[1].data(), fakeDataSize, bPos);
  n = 0;
  while (n < fakeDataSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][n], &data[0][n] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][n], &data[1][n] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    n += blockSize;
  }
  _LOG << "====" << std::endl;
  _LOG << "Offset: " << oas.getOffset() << std::endl;
  ENSURE((float)bPos / 2.0f == oas.getOffset(), "Half sample offset calculation failed");
  // Test half-sample delay
  {
    std::fill(begin(data[0]), end(data[0]), -1);
    std::fill(begin(data[1]), end(data[1]), -1);
    for (int i = 0; i < 10; i += 2) {
      data[0][i] = (sample_t)(i / 2) * 2;
      data[0][i + 1] = (sample_t)(i / 2) * 2;
      data[1][i + bPos] = (sample_t)(i) + 1;
      data[1][i + bPos + 1] = (sample_t)(i) + 1;
    }
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][0], &data[0][0] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][0], &data[1][0] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    sample_t *oPtr1 = ((sample_t **)out[0].getSamples().data())[0];
    sample_t *oPtr2 = ((sample_t **)out[1].getSamples().data())[0];
    ENSURE(oPtr1[bPos + kClickDelay] == 1, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay] == 1, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 1] == 1, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 1] == 1, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 2] == 3, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 2] == 3, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 3] == 3, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 3] == 3, "Fractional delay test failed");
    _LOG << "Fractional delay (half sample): OK" << std::endl;
  }
  _LOG << std::endl;
#endif

#if 1
  _LOG << "==== Testing fake data: Blanks in one channel (will reset) ====" << std::endl;
  // Generate blanks in one channel
  std::fill(begin(data[0]), end(data[0]), -1);
  std::fill(begin(data[1]), end(data[1]), -1);
  genData(data[0].data(), fakeDataSize, 0);
  n = 0;
  while (n < fakeDataSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][n], &data[0][n] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][n], &data[1][n] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    n += blockSize;
  }
  // Read real data until blank search times out
  f.seekg(0x2c, std::ios::beg);  // Start of audio data
  while (n < realDataSize - blockSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    readData((sample_t *)block1[0], (sample_t *)block2[0], blockSize);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    n += blockSize;
    sCount += blockSize;
  }
  _LOG << "====" << std::endl;
  _LOG << "Offset: " << oas.getOffset() << std::endl;
  ENSURE((float)bPos / 2.0f == oas.getOffset(), "Offset should not change if search times out");
  _LOG << std::endl;
#endif

#if 1
  _LOG << "==== Testing fake data: Blanks, full sample offset and latency ====" << std::endl;
  // Test full-sample detection on synthetic data
  bPos = 160;
  genData(data[0].data(), fakeDataSize, bPos);
  genData(data[1].data(), fakeDataSize, 0);
  n = 0;
  while (n < fakeDataSize) {
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][n], &data[0][n] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][n], &data[1][n] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    n += blockSize;
  }
  _LOG << "====" << std::endl;
  _LOG << "Offset: " << oas.getOffset() << std::endl;
  ENSURE((float)bPos / 2.0f == oas.getOffset(), "Whole sample offset calculation failed");
  // Test full sample delay
  {
    std::fill(begin(data[0]), end(data[0]), -1);
    std::fill(begin(data[1]), end(data[1]), -1);
    for (int i = 0; i < 10; i += 2) {
      data[1][i] = (sample_t)(i / 2) * 2;
      data[1][i + 1] = (sample_t)(i / 2) * 2;
      data[0][i + bPos] = (sample_t)(i) + 1;
      data[0][i + bPos + 1] = (sample_t)(i) + 1;
    }
    // To check processing delay, we put a value somewhere, and make sure
    // it comes out delayed by processing delay.
    int ot = 20;
    data[1][ot] = 12345;
    std::vector<Audio::Samples> in;
    block1[0] = new uint8_t[blockSize * sizeof(sample_t)];
    block2[0] = new uint8_t[blockSize * sizeof(sample_t)];
    std::copy(&data[0][0], &data[0][0] + blockSize, (sample_t *)block1[0]);
    std::copy(&data[1][0], &data[1][0] + blockSize, (sample_t *)block2[0]);
    Audio::Samples s1(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block1, blockSize / 2);
    Audio::Samples s2(Audio::SamplingRate::SR_44100, Audio::SamplingDepth::INT16, Audio::ChannelLayout::STEREO, 0,
                      block2, blockSize / 2);
    in.push_back(std::move(s1));
    in.push_back(std::move(s2));
    oas.process(in, out);
    sample_t *oPtr1 = ((sample_t **)out[0].getSamples().data())[0];
    sample_t *oPtr2 = ((sample_t **)out[1].getSamples().data())[0];
    ENSURE(oPtr1[bPos + kClickDelay] == 1, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay] == 0, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 1] == 1, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 1] == 0, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 2] == 3, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 2] == 2, "Fractional delay test failed");
    ENSURE(oPtr1[bPos + kClickDelay + 3] == 3, "Fractional delay test failed");
    ENSURE(oPtr2[bPos + kClickDelay + 3] == 2, "Fractional delay test failed");
    _LOG << "Fractional delay (whole sample): OK" << std::endl;
    // Make sure processing delay is correct. We places out sample at `ot`,
    // so it should now be at `ot` + procDelay*2.
    _LOG << "Processing latency: " << oas.getProcessingDelay() << std::endl;
    ENSURE(oPtr2[ot + int(2.f * oas.getProcessingDelay())] == 12345, "Get processing delay test failed");
  }
  _LOG << std::endl;
#endif

  return 0;
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "No test file given" << std::endl;
    return 17;
  } else {
    std::cout << "Using " << argv[1] << " for test" << std::endl;
  }
  return VideoStitch::Testing::testAudioSync(argv[1]);
}

#endif  // !defined(__clang_analyzer__)
