// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../audio/converter.hpp"

#include "gpu/testing.hpp"

namespace VideoStitch {
namespace Testing {

std::vector<std::vector<audioSample_t>> makeSamples(size_t nChannels) {
  std::vector<std::vector<audioSample_t>> samples;
  std::vector<audioSample_t> tmp;
  tmp.push_back(-1.0);
  tmp.push_back(0.0);
  tmp.push_back(1.0);
  for (size_t c = 0; c < nChannels; c++) {
    samples.push_back(tmp);
  }
  return samples;
}

void convDBLToINT16() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT16, nChannels, nFrames);
  ac->convert(samples);
  int16_t *outData = (int16_t *)ac->getdata();
  ENSURE_EQ((int16_t)(-32768), outData[0]); /*  -1.0   */
  ENSURE_EQ((int16_t)(-32768), outData[1]); /*  -1.0   */
  ENSURE_EQ((int16_t)0, outData[2]);        /*  0.0   */
  ENSURE_EQ((int16_t)0, outData[3]);        /*  0.0   */
  ENSURE_EQ((int16_t)(32767), outData[4]);  /*  1.0   */
  ENSURE_EQ((int16_t)(32767), outData[5]);  /*  1.0   */
  delete ac;
}

void convDBLToINT16_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT16_P, nChannels, nFrames);
  ac->convert(samples);
  int16_t *outData = (int16_t *)ac->getdata();
  ENSURE_EQ((int16_t)(-32768), outData[0]); /*  -1.0   */
  ENSURE_EQ((int16_t)(0), outData[1]);      /*  0.0   */
  ENSURE_EQ((int16_t)32767, outData[2]);    /*  1.0   */
  ENSURE_EQ((int16_t)-32768, outData[3]);   /*  -1.0   */
  ENSURE_EQ((int16_t)(0), outData[4]);      /*  0.0   */
  ENSURE_EQ((int16_t)(32767), outData[5]);  /*  1.0   */
  delete ac;
}

void convDBLToUINT8() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::UINT8, nChannels, nFrames);
  ac->convert(samples);
  uint8_t *outData = (uint8_t *)ac->getdata();
  ENSURE_EQ((uint8_t)(0), outData[0]);   /*  -1.0   */
  ENSURE_EQ((uint8_t)(0), outData[1]);   /*  -1.0   */
  ENSURE_EQ((uint8_t)128, outData[2]);   /*  0.0   */
  ENSURE_EQ((uint8_t)128, outData[3]);   /*  0.0   */
  ENSURE_EQ((uint8_t)(255), outData[4]); /*  1.0   */
  ENSURE_EQ((uint8_t)(255), outData[5]); /*  1.0   */
  delete ac;
}

void convDBLToUINT8_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::UINT8_P, nChannels, nFrames);
  ac->convert(samples);
  uint8_t *outData = (uint8_t *)ac->getdata();
  ENSURE_EQ((uint8_t)0, outData[0]);   /*  -1.0   */
  ENSURE_EQ((uint8_t)128, outData[1]); /*  -1.0   */
  ENSURE_EQ((uint8_t)255, outData[2]); /*  0.0   */
  ENSURE_EQ((uint8_t)0, outData[3]);   /*  0.0   */
  ENSURE_EQ((uint8_t)128, outData[4]); /*  1.0   */
  ENSURE_EQ((uint8_t)255, outData[5]); /*  1.0   */
  delete ac;
}

int32_t getInt24Val(uint8_t *src) {
  int32_t y = 0;
  uint8_t *yy = (uint8_t *)&y;
  yy[0] = src[0];
  yy[1] = src[1];
  yy[2] = src[2];
  return y;
}

void convDBLToINT24() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT24, nChannels, nFrames);
  ac->convert(samples);
  uint8_t *outData = (uint8_t *)ac->getdata();
  ENSURE_EQ((int32_t)(0x800000), getInt24Val(outData));      /*  -1.0   */
  ENSURE_EQ((int32_t)(0x800000), getInt24Val(outData + 3));  /*  -1.0   */
  ENSURE_EQ((int32_t)0, getInt24Val(outData + 6));           /*   0.0   */
  ENSURE_EQ((int32_t)0, getInt24Val(outData + 9));           /*   0.0   */
  ENSURE_EQ((int32_t)(0x7fffff), getInt24Val(outData + 12)); /*   1.0   */
  ENSURE_EQ((int32_t)(0x7fffff), getInt24Val(outData + 15)); /*   1.0   */
  delete ac;
}

void convDBLToINT24_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT24_P, nChannels, nFrames);
  ac->convert(samples);
  uint8_t *outData = (uint8_t *)ac->getdata();
  ENSURE_EQ((int32_t)(0x800000), getInt24Val(outData));      /*  -1.0   */
  ENSURE_EQ((int32_t)0, getInt24Val(outData + 3));           /*   0.0   */
  ENSURE_EQ((int32_t)(0x7fffff), getInt24Val(outData + 6));  /*   1.0   */
  ENSURE_EQ((int32_t)(0x800000), getInt24Val(outData + 9));  /*  -1.0   */
  ENSURE_EQ((int32_t)0, getInt24Val(outData + 12));          /*   0.0   */
  ENSURE_EQ((int32_t)(0x7fffff), getInt24Val(outData + 15)); /*   1.0   */
  delete ac;
}

void convDBLToINT32() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT32, nChannels, nFrames);
  ac->convert(samples);
  int32_t *outData = (int32_t *)ac->getdata();
  ENSURE_EQ((int32_t)(-2147483648), outData[0]); /*  -1.0   */
  ENSURE_EQ((int32_t)(-2147483648), outData[1]); /*  -1.0   */
  ENSURE_EQ((int32_t)0, outData[2]);             /*   0.0   */
  ENSURE_EQ((int32_t)0, outData[3]);             /*   0.0   */
  ENSURE_EQ((int32_t)(2147483647), outData[4]);  /*   1.0   */
  ENSURE_EQ((int32_t)(2147483647), outData[5]);  /*   1.0   */
  delete ac;
}

void convDBLToINT32_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::INT32_P, nChannels, nFrames);
  ac->convert(samples);
  int32_t *outData = (int32_t *)ac->getdata();
  ENSURE_EQ((int32_t)(-2147483648), outData[0]); /*  -1.0   */
  ENSURE_EQ((int32_t)(0), outData[1]);           /*   0.0   */
  ENSURE_EQ((int32_t)(2147483647), outData[2]);  /*   1.0   */
  ENSURE_EQ((int32_t)-2147483648, outData[3]);   /*  -1.0   */
  ENSURE_EQ((int32_t)(0), outData[4]);           /*   0.0   */
  ENSURE_EQ((int32_t)(2147483647), outData[5]);  /*   1.0   */
  delete ac;
}

void convDBLToFLT() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::FLT, nChannels, nFrames);
  ac->convert(samples);
  float *outData = (float *)ac->getdata();
  ENSURE_EQ((float)(-1.0), outData[0]); /*  -1.0   */
  ENSURE_EQ((float)(-1.0), outData[1]); /*  -1.0   */
  ENSURE_EQ((float)0., outData[2]);     /*   0.0   */
  ENSURE_EQ((float)0., outData[3]);     /*   0.0   */
  ENSURE_EQ((float)(1.0), outData[4]);  /*   1.0   */
  ENSURE_EQ((float)(1.0), outData[5]);  /*   1.0   */
  delete ac;
}

void convDBLToFLT_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::FLT_P, nChannels, nFrames);
  ac->convert(samples);
  float *outData = (float *)ac->getdata();
  ENSURE_EQ((float)(-1.0), outData[0]); /*  -1.0   */
  ENSURE_EQ((float)0., outData[1]);     /*   0.0   */
  ENSURE_EQ((float)(1.0), outData[2]);  /*   1.0   */
  ENSURE_EQ((float)(-1.0), outData[3]); /*  -1.0   */
  ENSURE_EQ((float)0., outData[4]);     /*   0.0   */
  ENSURE_EQ((float)(1.0), outData[5]);  /*   1.0   */
  delete ac;
}

void convDBLToDBL() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::DBL, nChannels, nFrames);
  ac->convert(samples);
  double *outData = (double *)ac->getdata();
  ENSURE_EQ((double)(-1.0), outData[0]); /*  -1.0   */
  ENSURE_EQ((double)(-1.0), outData[1]); /*  -1.0   */
  ENSURE_EQ((double)0., outData[2]);     /*   0.0   */
  ENSURE_EQ((double)0., outData[3]);     /*   0.0   */
  ENSURE_EQ((double)(1.0), outData[4]);  /*   1.0   */
  ENSURE_EQ((double)(1.0), outData[5]);  /*   1.0   */
  delete ac;
}

void convDBLToDBL_P() {
  std::vector<std::vector<audioSample_t>> samples = makeSamples(2);
  size_t nChannels = samples.size();
  size_t nFrames = samples[0].size();
  Audio::AudioConverter *ac = Audio::AudioConverter::create(Audio::SamplingDepth::DBL_P, nChannels, nFrames);
  ac->convert(samples);
  double *outData = (double *)ac->getdata();
  ENSURE_EQ((double)(-1.0), outData[0]); /*  -1.0   */
  ENSURE_EQ((double)0., outData[1]);     /*   0.0   */
  ENSURE_EQ((double)(1.0), outData[2]);  /*   1.0   */
  ENSURE_EQ((double)(-1.0), outData[3]); /*  -1.0   */
  ENSURE_EQ((double)0., outData[4]);     /*   0.0   */
  ENSURE_EQ((double)(1.0), outData[5]);  /*   1.0   */
  delete ac;
}

}  // namespace Testing
}  // namespace VideoStitch

int main(/*int argc, char **argv*/) {
  std::cout << "RUN Test convert DOUBLE to INT16" << std::endl;
  VideoStitch::Testing::convDBLToINT16();
  std::cout << "Test convert DOUBLE to INT16 : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to INT16_P " << std::endl;
  VideoStitch::Testing::convDBLToINT16_P();
  std::cout << "Test convert DOUBLE to INT16_P : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to UINT8 " << std::endl;
  VideoStitch::Testing::convDBLToUINT8();
  std::cout << "Test convert DOUBLE to UINT8 : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to UINT8_P " << std::endl;
  VideoStitch::Testing::convDBLToUINT8_P();
  std::cout << "Test convert DOUBLE to UINT8_P : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to INT24 " << std::endl;
  VideoStitch::Testing::convDBLToINT24();
  std::cout << "Test convert DOUBLE to INT24 : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to INT24_P " << std::endl;
  VideoStitch::Testing::convDBLToINT24_P();
  std::cout << "Test convert DOUBLE to INT24_P : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to INT32 " << std::endl;
  VideoStitch::Testing::convDBLToINT32();
  std::cout << "Test convert DOUBLE to INT32 : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to INT32_P " << std::endl;
  VideoStitch::Testing::convDBLToINT32_P();
  std::cout << "Test convert DOUBLE to INT32_P : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to FLT " << std::endl;
  VideoStitch::Testing::convDBLToFLT();
  std::cout << "Test convert DOUBLE to FLT : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to FLT_P " << std::endl;
  VideoStitch::Testing::convDBLToFLT_P();
  std::cout << "Test convert DOUBLE to FLT_P : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to DBL " << std::endl;
  VideoStitch::Testing::convDBLToDBL();
  std::cout << "Test convert DOUBLE to DBL : PASSED\n" << std::endl;

  std::cout << "RUN Test convert DOUBLE to DBL_P " << std::endl;
  VideoStitch::Testing::convDBLToDBL_P();
  std::cout << "Test convert DOUBLE to DBL_P : PASSED\n" << std::endl;
}
