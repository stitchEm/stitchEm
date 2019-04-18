// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveaudioprocessordelay.hpp"

#include "libvideostitch/ptv.hpp"

#include "widgets/audioprocessors/delayprocessorwidget.hpp"

LiveAudioProcessorDelay::LiveAudioProcessorDelay(const VideoStitch::Ptv::Value *config) : LiveAudioProcessFactory() {
  type = VideoStitch::AudioProcessors::ProcessorEnum::DELAY;
  if (config && config->has("delay")) {
    delay = config->has("delay")->asInt();
  }
}

VideoStitch::Ptv::Value *LiveAudioProcessorDelay::serialize() const {
  VideoStitch::Ptv::Value *value = VideoStitch::Ptv::Value::emptyObject();
  value->get("delay")->asInt() = delay;
  return value;
}

QPixmap LiveAudioProcessorDelay::getIcon() const { return QPixmap(""); }

QString LiveAudioProcessorDelay::getDescription() const { return QString(tr("A delay value for the audio input")); }

AudioProcessorConfigurationWidget *LiveAudioProcessorDelay::createConfigurationWidget(QWidget *parent) {
  return new DelayProcessorWidget(this, parent);
}

void LiveAudioProcessorDelay::serializeParameters(VideoStitch::Ptv::Value *value) {
  if (value) {
    value->push("delay", VideoStitch::Ptv::Value::intObject(delay));
  }
}

int LiveAudioProcessorDelay::getDelay() const { return delay; }

void LiveAudioProcessorDelay::setDelay(const int value) { delay = value; }
