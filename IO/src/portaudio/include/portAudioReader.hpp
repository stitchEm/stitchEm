// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <condition_variable>
#include <string>
#include <queue>
#include <thread>
#include "portaudio.h"
#include "libvideostitch/input.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/circularBuffer.hpp"

namespace VideoStitch {
namespace Input {

/**
 * PortAudio audio reader
 */

class PortAudioReader : public AudioReader {
 public:
  struct paDevice {
    PaStream* stream = nullptr;
    PaStreamParameters params;
    const PaDeviceInfo* devInfo = nullptr;
    double sampleRate;
    mtime_t offset;
  };

  static PortAudioReader* create(readerid_t id, const Ptv::Value* config);
  static bool handles(const Ptv::Value* config);

  virtual ~PortAudioReader();

  ReadStatus readSamples(size_t nbSamples, Audio::Samples&);
  Status seekFrame(mtime_t);
  size_t available();
  bool eos();

 private:
  PortAudioReader(readerid_t id, paDevice* dev);

  void startStream();

  // See portaudio.h
  static int paCaptureCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
                               const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
                               void* userData);

  bool stopping = false;
  paDevice* dev;
  std::mutex m;
  CircularBuffer<float> audioData;
  mtime_t audioDataTimestamp;
  mtime_t audioDataTimestampLast;
  size_t audioBufferSize;
  bool audioBufferOverflow;

};  // class PortAudioReader

}  // namespace Input
}  // namespace VideoStitch
