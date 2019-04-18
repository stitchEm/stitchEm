// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGEXPOSUREWIDGET_HPP
#define CONFIGEXPOSUREWIDGET_HPP

#include "ui_configexposurewidget.h"
#include "videostitcher/liveprojectdefinition.hpp"
#include "generic/backgroundcontainer.hpp"

class ConfigExposureWidget : public QWidget, public Ui::ConfigExposureWidgetClass {
  Q_OBJECT
 public:
  explicit ConfigExposureWidget(QWidget* const parent, LiveProjectDefinition* projectDefinition);
  ~ConfigExposureWidget();

  void updateEditability(bool outputIsActivated, bool algorithmIsActivated);

 protected:
  virtual void reactToChangedProject();
  virtual void reactToClearedProject();
  virtual void save();
  virtual void load();

 private slots:
  void onWidgetClosed();
  void onConfigurationChanged();

 private:
  void resetAnchorValues();
  BackgroundContainer* background;
  LiveProjectDefinition* projectDefinition;
};

#endif  // CONFIGEXPOSUREWIDGET_HPP
