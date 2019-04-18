// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iconfigurationvalue.hpp"

namespace Ui {
class AudioInputConfigurationWidget;
}

struct AudioConfiguration;
class PluginsController;

class AudioInputConfigurationWidget : public IConfigurationCategory {
  Q_OBJECT

 public:
  explicit AudioInputConfigurationWidget(QWidget* parent = nullptr);
  ~AudioInputConfigurationWidget();

  void setPluginsController(const PluginsController* newPluginsController);

  AudioConfiguration getAudioConfiguration() const;

 protected:
  virtual void reactToChangedProject();

 private:
  void setAudioConfiguration(const AudioConfiguration& config);
  void updateVisibility(QString device);
  QString getTextForNoDevice() const;
  QString getTextForProcedural() const;

 private slots:
  void updateDeviceSupportedValues();

 private:
  QScopedPointer<Ui::AudioInputConfigurationWidget> ui;
  const PluginsController* pluginsController = nullptr;
};
