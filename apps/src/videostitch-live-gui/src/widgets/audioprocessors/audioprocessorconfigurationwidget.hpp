// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iconfigurationvalue.hpp"

class LiveAudioProcessFactory;

class AudioProcessorConfigurationWidget : public IConfigurationCategory {
  Q_OBJECT
 public:
  explicit AudioProcessorConfigurationWidget(QWidget* parent = nullptr);
  virtual ~AudioProcessorConfigurationWidget() {}
  virtual LiveAudioProcessFactory* getConfiguration() const = 0;
  virtual void loadConfiguration() = 0;
 signals:
  void notifyConfigurationChanged(LiveAudioProcessFactory* config);
};
