// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_configurationoutputhdd.h"

#include "outputconfigurationwidget.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

class LiveOutputFile;

class ConfigurationOutputHDD : public OutputConfigurationWidget, public Ui::ConfigurationOutputHDDClass {
  Q_OBJECT

 public:
  // Does not take ownership of the live output
  explicit ConfigurationOutputHDD(LiveOutputFile* output, VideoStitch::OutputFormat::OutputFormatEnum type,
                                  QWidget* const parent = nullptr);
  virtual ~ConfigurationOutputHDD();

  virtual LiveOutputFactory* getOutput() const;

  virtual void toggleWidgetState();

 protected:
  virtual void reactToChangedProject();
  virtual void saveData();
  virtual void fillWidgetWithValue();

 private slots:
  void onCodecChanged(int index);
  void onFormatChanged(int index);
  void updateEncodingSettings();
  void onButtonBrowseClicked();
  void updateBitrateMaximum();

 private:
  QString getCurrentData(QComboBox* combo) const;
  void setCurrentIndexFromData(QComboBox* combo, const QString& format);

  LiveOutputFile* outputRef;
};
