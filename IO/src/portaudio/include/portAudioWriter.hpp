// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/circularBuffer.hpp"
#include "portaudio.h"
#include <condition_variable>
#include <string>
#include <queue>
#include <thread>

namespace VideoStitch {
namespace Output {

/**
 * PortAudio audio writer
 */

class PortAudioWriter : public AudioWriter {
 public:
  struct paDevice {
    PaStream* stream;
    PaStreamParameters params;
    const PaDeviceInfo* devInfo;
    double sampleRate;
    mtime_t offset;
  };

  static Potential<PortAudioWriter> create(const Ptv::Value* config,
                                           const VideoStitch::Plugin::VSWriterPlugin::Config& runtime);

  static bool handles(const Ptv::Value* config);

  virtual ~PortAudioWriter();

  void pushAudio(Audio::Samples& audioSamples) override;

 private:
  PortAudioWriter(const Plugin::VSWriterPlugin::Config& runtime, const paDevice paDev);

  void startStream();

  // See portaudio.h
  static int paPlayCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,
                            void* userData);

  static void paStreamFinished(void* userData);

  bool stopping;
  paDevice dev;
  std::mutex m;
  CircularBuffer<float> audioData;
  mtime_t audioDataTimestamp;
  mtime_t audioDataTimestampLast;
  size_t audioBufferSize;
  bool audioBufferOverflow;

};  // class PortAudioWriter

}  // namespace Output
}  // namespace VideoStitch
