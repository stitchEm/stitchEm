// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveaudio.hpp"

#include "guiconstants.hpp"
#include "liveoutputfactory.hpp"

#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"

LiveAudio::LiveAudio()
    : samplingRate(VideoStitch::Audio::getIntFromSamplingRate(VideoStitch::Audio::SamplingRate::SR_44100)),
      bitrate(DEFAULT_AUDIO_BITRATE),
      samplingFormat(QString()),  // Depends of the output
      audioCodec(VideoStitch::AudioHelpers::getStringFromCodec(VideoStitch::AudioHelpers::AudioCodecEnum::AAC)),
      channelLayout(VideoStitch::Audio::getStringFromChannelLayout(VideoStitch::Audio::STEREO)) {}

LiveAudio::~LiveAudio() {}

const QString LiveAudio::getAudioCodec() const { return audioCodec; }

int LiveAudio::getSamplingRate() const { return samplingRate; }

int LiveAudio::getBitrate() const { return bitrate; }

const QString LiveAudio::getSamplingFormat() const { return samplingFormat; }

const QString LiveAudio::getChannelLayout() const { return channelLayout; }

void LiveAudio::serializeIn(VideoStitch::Ptv::Value* config) const {
  if (!audioCodec.isEmpty()) {
    if (!GlobalController::getInstance().getController()->hasInputAudio()) {
      VideoStitch::Logger::get(VideoStitch::Logger::Warning)
          << "Audio configuration ignored as we don't have input audio." << std::endl;
      return;
    }

    config->get("audio_codec")->asString() = audioCodec.toStdString();
    config->get("sample_format")->asString() = samplingFormat.toStdString();
    config->get("sampling_rate")->asInt() = samplingRate;
    config->get("channel_layout")->asString() = channelLayout.toStdString();
    config->get("audio_bitrate")->asInt() = bitrate;
  }
}

void LiveAudio::setAudioCodec(const QString codecValue) { audioCodec = codecValue; }

void LiveAudio::setSamplingRate(const int samplingRateValue) { samplingRate = samplingRateValue; }

void LiveAudio::setSamplingFormat(const QString samplingDepthValue) { samplingFormat = samplingDepthValue; }

void LiveAudio::setChannelLayout(const QString channelLayoutValue) { channelLayout = channelLayoutValue; }

void LiveAudio::setBitrate(const int bitrateValue) { bitrate = bitrateValue; }

bool LiveAudio::isConfigured() const {
  return !samplingFormat.isEmpty() && !audioCodec.isEmpty() && !channelLayout.isEmpty() && samplingRate > 0 &&
         GlobalController::getInstance()
             .getController()
             ->hasInputAudio();  // If we don't have input audio -> output audio is not configured.
}

void LiveAudio::initialize(const VideoStitch::Ptv::Value* config, const LiveOutputFactory& output) {
  VideoStitch::Parse::populateInt("Ptv", *config, "sampling_rate", samplingRate, false);
  VideoStitch::Parse::populateInt("Ptv", *config, "audio_bitrate", bitrate, false);

  std::string codec;
  std::string sampleFormat;
  std::string layout;
  if (VideoStitch::Parse::populateString("Ptv", *config, "audio_codec", codec, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    audioCodec = QString::fromStdString(codec);
  }
  if (VideoStitch::Parse::populateString("Ptv", *config, "sample_format", sampleFormat, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    samplingFormat = QString::fromStdString(sampleFormat);
  } else {
    const VideoStitch::AudioHelpers::AudioCodecEnum audioCodecType =
        VideoStitch::AudioHelpers::getCodecFromString(audioCodec);
    const VideoStitch::Audio::SamplingDepth samplingDepth = output.getPreferredSamplingDepth(audioCodecType);
    samplingFormat = VideoStitch::Audio::getStringFromSamplingDepth(samplingDepth);
  }
  if (VideoStitch::Parse::populateString("Ptv", *config, "channel_layout", layout, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    channelLayout = QString::fromStdString(layout);
  }
}
