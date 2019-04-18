// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <iostream>

#include "orah4i2b.hpp"
#include "orah4i2b_test_data.h"

// Note: you cannot use DO_SPEED_TEST and an other test at the same time
#define DO_SPEED_TEST 0  // Dev only
// MATLAB_TEST add the capability of injecting a wave file into the O2B test
// This wave file format has to be 4 channels, 44100 Hz, int_16
#define DO_MATLAB_TEST 0  // Dev only
#define DO_ACCURACY_TEST 1

#if DO_SPEED_TEST
#include <chrono>
#if defined(NDEBUG)
static const int kMaxTime{60};
#else
static const int kMaxTime{600};
#endif
#endif
#if DO_MATLAB_TEST
#include <string.h>
#endif
#if DO_ACCURACY_TEST
#include <cmath>
// An error greater than 0.5 bits for 32-bit audio is for all intents
// and purposes equivalent to 0. Max error is thus 2^-32, or 2.33e-10.
static const double kMaxError{2.33e-10};
#endif

int main(int /* argc */, char** /* argv */) {
  VideoStitch::Orah4i::Orah4iToB* toB = nullptr;
  int blockSize = VideoStitch::Orah4i::get4iToBBlockSize();
  double* out = new double[blockSize * 4];
  int ret = 1;

#if DO_MATLAB_TEST
  // The wave file has to be 4 channels, 44100 Hz, int_16 to be understood by the test
  const std::string fileName = "yourinputfile.wav";
  FILE* inFile = fopen(fileName.c_str(), "rb");
  // This has been only tested for the files provided by Illusonic.
  // If you inject a file from other origins the "data" subchunk could be at a different offset.
  // For example it's at 74 for a file from FFMPEG.
  fseek(inFile, 36, SEEK_SET);
  char chunk[4];
  double inData[blockSize * 4];
  if (fread(chunk, sizeof(chunk), 1, inFile) != 1) {
    std::cerr << "Failed to read the wave file " << fileName << std::endl;
    goto out_fail;
  }

  if (strcmp(chunk, "data") != 0) {
    std::cerr << "Error while reading " << fileName << ". Failed to find data field in " << fileName << std::endl;
    goto out_fail;
  }

  uint32_t subchunksize;
  if (fread(&subchunksize, sizeof(subchunksize), 1, inFile) != 1) {
    goto out_fail;
  }
  std::cout << "chunk " << chunk << " of size " << subchunksize << std::endl;
#endif
  // Speed test
#if DO_SPEED_TEST
#define N_LOOPS 10000
  long long t = 0;
  toB = VideoStitch::Orah4i::Orah4iToB::create();
  if (!toB) {
    std::cerr << "[orah4i2b] Filed to initialize" << std::endl;
    goto out_fail;
  }
  for (int j = 0; j < N_LOOPS; j++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
      if (!toB->process((double*)&rdata[i * blockSize * 4], out)) {
        std::cerr << "[orah4i2b] Failed to process audio" << std::endl;
        goto out_fail;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now() - t0;
    t += std::chrono::duration_cast<std::chrono::microseconds>(t1).count();
  }
  t /= (N_LOOPS * 4);
  if (t > kMaxTime) {
    std::cerr << "[orah4i2b] Expected average block process time of less than " << kMaxTime << std::endl;
    goto out_fail;
  }
  std::cout << "SPEED TEST RESULT " << t << " us" << std::endl;
#endif

  // Accuracy test
#if DO_MATLAB_TEST
  if (toB) {
    delete toB;
  }
  toB = VideoStitch::Orah4i::Orah4iToB::create();
  if (!toB) {
    std::cerr << "[orah4i2b] Filed to initialize" << std::endl;
    goto out_fail;
  }
  // If remapping is needed you can use this to remap your channels
  int remap[4];
  remap[0] = 0;
  remap[1] = 1;
  remap[2] = 2;
  remap[3] = 3;
  for (int nbBlock = 0; nbBlock < 100; nbBlock++) {
    for (int s = 0; s < blockSize; s++) {
      for (int c = 0; c < 4; c++) {
        int16_t valint16;
        if (fread((char*)&valint16, sizeof(int16_t), 1, inFile) == 1) {
          if (valint16 > 0) {
            inData[s * 4 + remap[c]] = static_cast<double>(valint16) / 32767.;
          } else {
            inData[s * 4 + remap[c]] = static_cast<double>(valint16) / 32768.;
          }
        } else {
          std::cerr << "Fail to read input file " << fileName << std::endl;
        }
      }
    }

    if (!toB->process(inData, out)) {
      std::cerr << "[orah4i2b] Failed to process audio" << std::endl;
      goto out_fail;
    }
  }

#endif

#if DO_ACCURACY_TEST
  if (toB) {
    delete toB;
  }
  toB = VideoStitch::Orah4i::Orah4iToB::create();
  if (!toB) {
    std::cerr << "[orah4i2b] Filed to initialize" << std::endl;
    goto out_fail;
  }
  for (int i = 0; i < 4; i++) {
    if (!toB->process(&rdata[i * blockSize * 4], out)) {
      std::cerr << "[orah4i2b] Failed to process audio" << std::endl;
      goto out_fail;
    }
  }
  for (int i = 0; i < blockSize * 4; i++) {
    if (std::abs(out[i] - good_result[i]) > kMaxError) {
      std::cerr << "[orah4i2b] Expected delta < " << kMaxError << ", got " << (std::abs(out[i] - good_result[i]))
                << std::endl;
      goto out_fail;
    }
  }
#endif

  ret = 0;

out_fail:
  delete toB;
  delete[] out;
  return ret;
}
