// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../gpu/testing.hpp"

#include "libvideostitch/audioWav.hpp"

namespace VideoStitch {
namespace Testing {
using namespace Audio;

void copyWavFile(const std::string &in, const std::string &out) {
  std::cout << "Copy " << in << " in " << out << std::endl;
  WavReader inwav(in);
  int nChannels = inwav.getChannels();
  ChannelLayout layout = MONO;

  switch (nChannels) {
    case 1:
      layout = MONO;
      break;
    case 2:
      layout = STEREO;
      break;
    default:
      std::cout << "nb of channels " << nChannels << " not managed" << std::endl;
      assert(false);
  }

  WavWriter outwav(out.c_str(), layout, inwav.getSampleRate());
  AudioBlock block(layout);
  inwav.step(block);
  outwav.step(block);
  outwav.close();
}

void compareWavFile(const std::string &a, const std::string &b, const double eps, size_t start = 0) {
  WavReader wra(a), wrb(b);
  AudioBlock bufa, bufb;
  wra.step(bufa);
  wrb.step(bufb);
  ENSURE_EQ(bufa.size(), bufb.size());
  ENSURE_EQ(bufa[0].size(), bufb[0].size());

  AudioBlock::size_type nbChannels = bufa.size();
  size_t nbSamples = bufa[0].size();

  double x, y;
  for (AudioBlock::size_type i = 0; i < nbChannels; i++) {
    for (size_t j = start; j < nbSamples; j++) {
      x = bufa[i].data()[j];
      y = bufb[i].data()[j];
      ENSURE_APPROX_EQ(x, y, eps);
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch
