// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGCALIBRATIONWIDGET_HPP
#define CONFIGCALIBRATIONWIDGET_HPP

#include "ui_configcalibrationwidget.h"
#include "iappsettings.hpp"

class ConfigCalibrationWidget : public IAppSettings, public Ui::ConfigCalibrationWidgetClass {
  Q_OBJECT
 public:
  explicit ConfigCalibrationWidget(QWidget* const parent = nullptr);
  ~ConfigCalibrationWidget();

  void updateEditability(bool outputIsActivated, bool algorithmIsActivated);

  virtual void load();
  virtual void save();

 public slots:
  void onButtonBrowseClicked();
};

#endif  // CONFIGCALIBRATIONWIDGET_HPP
