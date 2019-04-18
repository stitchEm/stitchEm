// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputaja.hpp"

#include "liveaudio.hpp"
#include "configurations/ajaoutputconfiguration.hpp"
#include "plugin/pluginscontroller.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

#include "libvideostitch/parse.hpp"

LiveOutputAJA::LiveOutputAJA(const VideoStitch::Ptv::Value* config,
                             const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveOutputSDI(config, type), device(0), channel(1) {
  fillOutputValues(config);
}

QList<VideoStitch::Audio::SamplingDepth> LiveOutputAJA::getSupportedSamplingDepths(
    const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const {
  Q_UNUSED(audioCodec);
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  return PluginsController::listAudioSamplingFormats(cardName);
}

bool LiveOutputAJA::getAudioIsEnabled() const { return audioIsEnabled; }

VideoStitch::Ptv::Value* LiveOutputAJA::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = "aja";
  value->get("filename")->asString() = deviceName.toStdString();
  value->get("device_display_name")->asString() = deviceDisplayName.toStdString();
  value->get("width")->asInt() = displayMode.width;
  value->get("height")->asInt() = displayMode.height;
  value->get("interleaved")->asBool() = displayMode.interleaved;
  value->get("psf")->asBool() = displayMode.psf;
  VideoStitch::Ptv::Value* fps = VideoStitch::Ptv::Value::emptyObject();
  fps->get("num")->asInt() = displayMode.framerate.num;
  fps->get("den")->asInt() = displayMode.framerate.den;
  value->push("frame_rate", fps);
  value->get("device")->asInt() = nameToDevice(deviceName);
  value->get("channel")->asInt() = nameToChannel(deviceName);
  value->get("audio")->asBool() = audioIsEnabled;
  value->get("offset_x")->asInt() = offset_x;
  value->get("offset_y")->asInt() = offset_y;
  return value;
}

OutputConfigurationWidget* LiveOutputAJA::createConfigurationWidget(QWidget* const parent) {
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  AjaOutputConfiguration* ajaConfig = new AjaOutputConfiguration(this, parent);
  ajaConfig->setSupportedDisplayModes(PluginsController::listDisplayModes(cardName, QStringList() << deviceName));
  return ajaConfig;
}

LiveOutputFactory::PanoSizeChange LiveOutputAJA::supportPanoSizeChange(int newWidth, int newHeight) const {
  if ((newWidth <= displayMode.width) && (newHeight <= displayMode.height)) {
    return PanoSizeChange::Supported;
  }
  auto reqDisplayMode = VideoStitch::Plugin::DisplayMode(newWidth, newHeight, displayMode.interleaved,
                                                         displayMode.framerate, displayMode.psf);
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  for (auto candidateDisplayMode : PluginsController::listDisplayModes(cardName, QStringList() << deviceName)) {
    if (candidateDisplayMode.canSupport(reqDisplayMode)) {
      return PanoSizeChange::SupportedWithUpdate;
    }
  }

  return PanoSizeChange::NotSupported;
}

QString LiveOutputAJA::getPanoSizeChangeDescription(int newWidth, int newHeight) const {
  auto reqDisplayMode = VideoStitch::Plugin::DisplayMode(newWidth, newHeight, displayMode.interleaved,
                                                         displayMode.framerate, displayMode.psf);
  QString desc = QString("%0 (%1)\n").arg(getOutputTypeDisplayName()).arg(getOutputDisplayName());
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  for (auto candidateDisplayMode : PluginsController::listDisplayModes(cardName, QStringList() << deviceName)) {
    if (candidateDisplayMode.canSupport(reqDisplayMode)) {
      desc += tr("- Display size: %0x%1").arg(candidateDisplayMode.width).arg(candidateDisplayMode.height);
      if (displayMode.framerate != candidateDisplayMode.framerate) {
        desc += tr("\n- Frame rate: %0/%1 fps")
                    .arg(candidateDisplayMode.framerate.num)
                    .arg(candidateDisplayMode.framerate.den);
      }
      return desc;
    }
  }
  desc += tr("- No compatible Display mode");
  return desc;
}

void LiveOutputAJA::updateForPanoSizeChange(int newWidth, int newHeight) {
  auto reqDisplayMode = VideoStitch::Plugin::DisplayMode(newWidth, newHeight, displayMode.interleaved,
                                                         displayMode.framerate, displayMode.psf);
  QString cardName = VideoStitch::InputFormat::getStringFromEnum(VideoStitch::InputFormat::InputFormatEnum::AJA);
  for (auto candidateDisplayMode : PluginsController::listDisplayModes(cardName, QStringList() << deviceName)) {
    if (candidateDisplayMode.canSupport(reqDisplayMode)) {
      displayMode = candidateDisplayMode;
      break;
    }
  }
}

void LiveOutputAJA::setAudioIsEnabled(bool audio) { audioIsEnabled = audio; }

void LiveOutputAJA::fillOutputValues(const VideoStitch::Ptv::Value* config) {
  VideoStitch::Parse::populateInt("Ptv", *config, "device", device, true);
  VideoStitch::Parse::populateInt("Ptv", *config, "channel", channel, true);
  VideoStitch::Parse::populateBool("Ptv", *config, "audio", audioIsEnabled, false);
}

QString LiveOutputAJA::toReadableName(const QString name) const {
  return QString(tr("Device %0 channel %1").arg(name.at(0)).arg(name.at(1)));
}

int LiveOutputAJA::nameToDevice(const QString name) const { return name.left(1).toInt(); }

int LiveOutputAJA::nameToChannel(const QString name) const { return name.right(1).toInt(); }
