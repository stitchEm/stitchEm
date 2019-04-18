// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGURATIONOUTPUTSDI_HPP
#define CONFIGURATIONOUTPUTSDI_HPP

#include "ui_configurationoutputsdi.h"
#include "outputconfigurationwidget.hpp"
#include "utils/displaymode.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

class LiveOutputSDI;
class LiveProjectDefinition;

class ConfigurationOutputSDI : public OutputConfigurationWidget, public Ui::ConfigurationOutputSDIClass {
  Q_OBJECT

 public:
  // Does not take ownership of the live output
  explicit ConfigurationOutputSDI(LiveOutputSDI* output, VideoStitch::OutputFormat::OutputFormatEnum type,
                                  QWidget* const parent = nullptr);
  ~ConfigurationOutputSDI();

  virtual void toggleWidgetState();
  void setSupportedDisplayModes(std::vector<VideoStitch::Plugin::DisplayMode> displayModes);
  virtual LiveOutputFactory* getOutput() const;

 protected:
  virtual void reactToChangedProject();
  virtual void saveData();
  virtual void fillWidgetWithValue();

 private:
  LiveOutputSDI* outputRef;
};

#endif  // CONFIGURATIONOUTPUTSDI_HPP
