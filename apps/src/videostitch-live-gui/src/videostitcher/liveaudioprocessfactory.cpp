// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveaudioprocessfactory.hpp"

#include "liveaudioprocessordelay.hpp"
#include "videostitcher/audioconfiguration.hpp"

#include <QVector>

LiveAudioProcessFactory* LiveAudioProcessFactory::createProcessor(
    const VideoStitch::Ptv::Value* config, const VideoStitch::AudioProcessors::ProcessorEnum& type) {
  switch (type) {
    case VideoStitch::AudioProcessors::ProcessorEnum::DELAY:
      return new LiveAudioProcessorDelay(config);
    default:
      return nullptr;
  }
}

QVector<QPair<QString, QString>> LiveAudioProcessFactory::getAvailableProcessors() {
  QVector<QPair<QString, QString>> availableProcessors;
  availableProcessors.append(qMakePair(QString("Audio delay"), QString("delay")));
  return availableProcessors;
}

VideoStitch::AudioProcessors::ProcessorEnum LiveAudioProcessFactory::getType() const { return type; }

const std::string& LiveAudioProcessFactory::getInputName() const { return inputName; }

std::unique_ptr<VideoStitch::Ptv::Value> LiveAudioProcessFactory::createDefaultConfig(
    const QString& inputName, const VideoStitch::AudioProcessors::ProcessorEnum& type) {
  Q_UNUSED(type);
  VideoStitch::Ptv::Value* params = VideoStitch::Ptv::Value::emptyObject();
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->push("input", VideoStitch::Ptv::Value::stringObject(inputName.toStdString()));
  params->asList().push_back(value);
  return std::unique_ptr<VideoStitch::Ptv::Value>(params);
}
