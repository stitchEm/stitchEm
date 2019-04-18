// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveaudioprocessfactory.hpp"

class LiveAudioProcessorDelay : public LiveAudioProcessFactory {
 public:
  explicit LiveAudioProcessorDelay(const VideoStitch::Ptv::Value* config);
  ~LiveAudioProcessorDelay() = default;

  QPixmap getIcon() const override;
  QString getDescription() const override;
  int getDelay() const;

  void setDelay(const int value);

  VideoStitch::Ptv::Value* serialize() const override;
  AudioProcessorConfigurationWidget* createConfigurationWidget(QWidget* parent) override;
  void serializeParameters(VideoStitch::Ptv::Value* value) override;

 private:
  int delay = 0;
};
