// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Ambisonic encoder tool to encode a wav file to B-format

#include <libvideostitch/audio.hpp>
#include <libvideostitch/audioWav.hpp>
#include <libvideostitch/ambisonic.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

namespace VideoStitch {
namespace Audio {
namespace {

void encodeInAmbisonic(const std::string &inWav, const std::string &outWav, AngularPosition inpos,
                       AmbisonicOrder order = AmbisonicOrder::FIRST_ORDER, AmbisonicNorm norm = AmbisonicNorm::SN3D) {
  std::cout << "Encode " << inWav << " to " << outWav << std::endl
            << " in B-format " << getStringFromAmbisonicOrder(order) << std::endl
            << " using normalisation " << getStringFromAmbisonicNorm(norm) << std::endl;

  int blockSize = getDefaultBlockSize();
  WavReader inReader(inWav.c_str());
  WavWriter outWriter(outWav.c_str(), getChannelLayoutFromAmbisonicOrder(order), inReader.getSampleRate());
  AmbEncoder ambEnc(order, norm);

  if (inReader.getLayout() == MONO) {
    std::cout << "Set the mono source position at azimuth " << inpos.az << " elevation " << inpos.el << " rad"
              << std::endl;
    ambEnc.setMonoSourcePosition(inpos);
  } else {
    std::cout << " Layout of the source " << getStringFromChannelLayout(inReader.getLayout()) << std::endl;
  }

  int nSamples = inReader.getnSamples(), nReadSamples = 0;
  int nSamplesToRead = nSamples;

  while (nSamplesToRead > 0) {
    AudioBlock inBlock(inReader.getLayout());
    AudioBlock outBlock(inReader.getLayout());
    if (nSamplesToRead >= blockSize) {
      nReadSamples = blockSize;
    } else if (nReadSamples > 0) {
      nReadSamples = nSamplesToRead;
    } else {
      break;
    }
    inReader.step(inBlock, nReadSamples);
    ambEnc.step(outBlock, inBlock);
    outWriter.step(outBlock);
    nSamplesToRead -= nReadSamples;
  }
  outWriter.close();
}
}  // namespace
}  // namespace Audio

}  // namespace VideoStitch

int main(int argc, char **argv) {
  std::string inWav, outWav;
  if (argc < 3) {
    std::cout << "Wrong usage of " << argv[0] << std::endl;
    std::cout << "Usage: " << argv[0] << " [input] [output] ([az] [el])" << std::endl;
    return 0;
  }

  inWav = argv[1];
  outWav = argv[2];

  double az = 0.;
  double el = 0.;
  if (argc == 5) {
    az = atof(argv[3]);
    el = atof(argv[4]);
  }

  VideoStitch::Audio::encodeInAmbisonic(inWav, outWav, {az, el});

  return 0;
}
