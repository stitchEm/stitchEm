// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Audio input def parser

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/audio.hpp"

namespace VideoStitch {
namespace Core {
using namespace Audio;

class AudioSourceDefinition::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

 private:
  friend class AudioSourceDefinition;
  readerid_t readerId;
  size_t channelId;
};

/**
 * Pimpl holder for AudioProcessortDef.
 */
class AudioProcessorDef::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

 private:
  friend class AudioProcessorDef;
  std::string name;
  std::unique_ptr<Ptv::Value> parameters;
};

/**
 * Pimpl holder for InputDefinition.
 */
class AudioInputDefinition::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

 private:
  friend class AudioInputDefinition;
  std::string name;
  bool isMaster;
  std::string layout;
  std::vector<std::unique_ptr<AudioSourceDefinition>> sources;
};

/**
 * Pimpl holder for AudioPipeDefinition.
 */
class AudioPipeDefinition::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

 private:
  friend class AudioPipeDefinition;
  std::string debugFolder;
  std::string selectedAudio;
  int blockSize;
  int samplingRate;
  bool hasVuMeter;
  std::vector<std::unique_ptr<AudioInputDefinition>> audioInputs;
  std::vector<std::unique_ptr<AudioMixDefinition>> audioMixes;
  std::vector<std::unique_ptr<AudioProcessorDef>> audioProcessors;
  std::unique_ptr<Audio::AmbisonicDecoderDef> ambDecCoef;
};

/**
 * Pimpl holder for AudioMixDefinition.
 */
class AudioMixDefinition::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

 private:
  friend class AudioMixDefinition;
  std::string name;
  std::vector<std::string> inputs;
};

}  // namespace Core
}  // namespace VideoStitch
