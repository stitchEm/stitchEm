// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveinputaja.hpp"
#include "libvideostitch/ptv.hpp"

LiveInputAJA::LiveInputAJA(const QString& name) : CaptureCardLiveInput(name) {
  // Default values for Aja
  displayMode = VideoStitch::Plugin::DisplayMode(1920, 1080, false, {30, 1});
}

LiveInputAJA::~LiveInputAJA() {}

VideoStitch::InputFormat::InputFormatEnum LiveInputAJA::getType() const {
  return VideoStitch::InputFormat::InputFormatEnum::AJA;
}

VideoStitch::Ptv::Value* LiveInputAJA::serialize() const {
  VideoStitch::Ptv::Value* input = CaptureCardLiveInput::serialize();
  input->get("reader_config")->get("device")->asInt() = nameToDevice(name);
  input->get("reader_config")->get("channel")->asInt() = nameToChannel(name);
  input->get("reader_config")->get("audio")->asBool() = audioIsEnabled;
  input->get("audio_enabled")->asBool() = audioIsEnabled;
  return input;
}

void LiveInputAJA::initializeWith(const VideoStitch::Ptv::Value* initializationInput) {
  CaptureCardLiveInput::initializeWith(initializationInput);

  const VideoStitch::Ptv::Value* reader_config = initializationInput->has("reader_config");
  if (reader_config->has("audio")) {
    audioIsEnabled = reader_config->has("audio")->asBool();
  }
}

void LiveInputAJA::setAudioIsEnabled(bool audio) { audioIsEnabled = audio; }

int LiveInputAJA::nameToDevice(const QString name) const { return name.left(1).toInt(); }

int LiveInputAJA::nameToChannel(const QString name) const { return name.right(1).toInt(); }
