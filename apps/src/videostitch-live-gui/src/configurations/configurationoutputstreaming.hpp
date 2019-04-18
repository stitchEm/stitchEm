// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGURATIONOUTPUTSTREAMING_HPP
#define CONFIGURATIONOUTPUTSTREAMING_HPP

#include "ui_configurationoutputstreaming.h"
#include "outputconfigurationwidget.hpp"

class LiveOutputRTMP;
class IStreamingServiceConfiguration;

class ConfigurationOutputStreaming : public OutputConfigurationWidget, public Ui::ConfigurationOutputStreaming {
  Q_OBJECT

 public:
  // Does not take ownership of the live output
  explicit ConfigurationOutputStreaming(LiveOutputRTMP* outputref, QWidget* const parent = nullptr);
  virtual ~ConfigurationOutputStreaming();

  virtual LiveOutputFactory* getOutput() const;

  virtual void toggleWidgetState();

 protected:
  virtual void updateAfterChangedMode();
  virtual void saveData();
  virtual void fillWidgetWithValue();
  virtual bool hasValidConfiguration() const;

 private slots:
  void updateEncodingSettings();
  void updateBitrateMaximum();
  void updateBitrateWidgetsVisibility(QString bitrateMode);
  void updatePresetWidgetsVisibility(QString encoder);
  void setBufferSizeToDefault();
  void updateMinBitrateMaximum();
  void setMinBitrateToDefault();
  void onBasicConfigurationComplete();
  void onBasicConfigurationCanceled();

 private:
  virtual void reactToChangedProject();

  LiveOutputRTMP* outputRef;
  IStreamingServiceConfiguration* streamingServiceConfig;
};

#endif  // CONFIGURATIONOUTPUTSTREAMING_HPP
