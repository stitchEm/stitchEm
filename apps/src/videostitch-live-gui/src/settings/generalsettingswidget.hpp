// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GENERALSETTINGSWIDGET_HPP
#define GENERALSETTINGSWIDGET_HPP

#include "ui_generalsettingswidget.h"
#include <QFrame>

class BackgroundContainer;

class GeneralSettingsWidget : public QFrame, public Ui::GeneralSettingsWidget {
  Q_OBJECT
 public:
  explicit GeneralSettingsWidget(QWidget* const parent = nullptr);
  virtual ~GeneralSettingsWidget();

 private slots:
  void onSettingsSaved();

 signals:
  void notifySettingsSaved();

 private:
  void showWarning(const bool show);
};

// GeneralSettingsDialog manages itself its lifecycle when we show it
class GeneralSettingsDialog : public QFrame {
  Q_OBJECT
 public:
  explicit GeneralSettingsDialog(QWidget* const parent = nullptr);
  virtual ~GeneralSettingsDialog();

 private slots:
  void onWidgetClosed();

 private:
  GeneralSettingsWidget* widget;
  BackgroundContainer* background;
};

#endif  // GENERALSETTINGSWIDGET_HPP
