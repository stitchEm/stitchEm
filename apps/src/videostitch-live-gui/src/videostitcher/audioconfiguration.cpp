// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioconfiguration.hpp"

#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"

AudioConfiguration::AudioConfiguration(const VideoStitch::Ptv::Value& readerConfig) {
  if (readerConfig.has("type") &&
      readerConfig.has("type")->asString() == "openAL") {  // Keep this for backwards compatibility
    inputName = readerConfig.has("device")->asString();
    nbAudioChannels = 2;
    audioRate = VideoStitch::Audio::SamplingRate::SR_48000;
  } else {
    type = readerConfig.has("type") ? readerConfig.has("type")->asString() : "";
    inputName = readerConfig.has("name") ? readerConfig.has("name")->asString() : "";
    nbAudioChannels = readerConfig.has("audio_channels") ? readerConfig.has("audio_channels")->asInt() : 0;
    if (readerConfig.has("sampling_rate")) {
      audioRate = VideoStitch::Audio::getSamplingRateFromInt(readerConfig.has("sampling_rate")->asInt());
    }
  }
}

bool AudioConfiguration::isValid() const {
  // If Aja, we don't need the nb of channels and the sampling rate
  return !inputName.empty() &&
         (isAja() || (nbAudioChannels > 0 && audioRate != VideoStitch::Audio::SamplingRate::SR_NONE));
}

bool AudioConfiguration::isAja() const {
  return VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA).toStdString() ==
         type;
}

VideoStitch::Ptv::Value* AudioConfiguration::serialize() const {
  VideoStitch::Ptv::Value* readerConfig = VideoStitch::Ptv::Value::emptyObject();
  readerConfig->get("type")->asString() = type;
  readerConfig->get("name")->asString() = inputName;
  readerConfig->get("audio_channels")->asInt() = nbAudioChannels;
  readerConfig->get("sampling_rate")->asInt() = VideoStitch::Audio::getIntFromSamplingRate(audioRate);

  VideoStitch::Ptv::Value* audioInput(VideoStitch::Ptv::Value::emptyObject());
  audioInput->push("reader_config", readerConfig);
  audioInput->get("video_enabled")->asBool() = false;
  audioInput->get("audio_enabled")->asBool() = true;
  return audioInput;
}
