// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_ajaoutputconfiguration.h"
#include "outputconfigurationwidget.hpp"
#include "utils/displaymode.hpp"

class LiveOutputAJA;
class LiveProjectDefinition;

class AjaOutputConfiguration : public OutputConfigurationWidget, public Ui::AjaOutputConfigurationClass {
  Q_OBJECT

 public:
  // Does not take ownership of the live output
  explicit AjaOutputConfiguration(LiveOutputAJA* output, QWidget* const parent = nullptr);
  ~AjaOutputConfiguration();

  virtual void toggleWidgetState();
  void setSupportedDisplayModes(std::vector<VideoStitch::Plugin::DisplayMode> displayModes);
  virtual LiveOutputFactory* getOutput() const;

 protected:
  virtual void reactToChangedProject();
  virtual void saveData();
  virtual void fillWidgetWithValue();

 private slots:
  void updateOffset();

 private:
  Q_DISABLE_COPY(AjaOutputConfiguration)

  LiveOutputAJA* outputRef;
};
