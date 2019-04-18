// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputdecklink.hpp"

#include "liveaudio.hpp"
#include "configurations/configurationoutputsdi.hpp"
#include "plugin/pluginscontroller.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch/parse.hpp"

LiveOutputDecklink::LiveOutputDecklink(const VideoStitch::Ptv::Value* config,
                                       const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveOutputSDI(config, type) {}

QList<VideoStitch::Audio::SamplingDepth> LiveOutputDecklink::getSupportedSamplingDepths(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  Q_UNUSED(audioCodec);
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::DECKLINK);
  return PluginsController::listAudioSamplingFormats(cardName);
}

VideoStitch::Ptv::Value* LiveOutputDecklink::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = "decklink";
  value->get("filename")->asString() = deviceName.toStdString();
  value->get("device_display_name")->asString() = deviceDisplayName.toStdString();
  value->get("width")->asInt() = displayMode.width;
  value->get("height")->asInt() = displayMode.height;
  value->get("interleaved")->asBool() = displayMode.interleaved;
  VideoStitch::Ptv::Value* fps = VideoStitch::Ptv::Value::emptyObject();
  fps->get("num")->asInt() = displayMode.framerate.num;
  fps->get("den")->asInt() = displayMode.framerate.den;
  value->push("frame_rate", fps);
  audioConfig->serializeIn(value);
  return value;
}

OutputConfigurationWidget* LiveOutputDecklink::createConfigurationWidget(QWidget* const parent) {
  QString cardName = VideoStitch::OutputFormat::getStringFromEnum(type);
  ConfigurationOutputSDI* decklinkConfig = new ConfigurationOutputSDI(this, type, parent);
  decklinkConfig->setSupportedDisplayModes(PluginsController::listDisplayModes(cardName, QStringList() << deviceName));
  return decklinkConfig;
}
