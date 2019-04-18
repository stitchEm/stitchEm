// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OCULUSSETTINGSWIDGET_HPP
#define OCULUSSETTINGSWIDGET_HPP

#include "iappsettings.hpp"

namespace Ui {
class OculusSettingsWidget;
}

class OculusSettingsWidget : public IAppSettings {
  Q_OBJECT

 public:
  explicit OculusSettingsWidget(QWidget* parent = nullptr);
  ~OculusSettingsWidget();

  virtual void load();
  virtual void save();

 private slots:
  void checkForChanges();

 private:
  QScopedPointer<Ui::OculusSettingsWidget> ui;
};

#endif  // OCULUSSETTINGSWIDGET_HPP
