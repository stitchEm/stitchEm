// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "ui_newoutputwidget.h"

class QLabel;
class NewOutputWidget : public QWidget, public Ui::NewOutputWidget {
  Q_OBJECT

 public:
  explicit NewOutputWidget(QWidget* const parent = nullptr);

  void insertDeviceItem(const QString displayName, const QString model, const QString pluginType, const bool isUsed);
  void clearDevices();

 signals:
  void notifyDevicesSelected(const QString displayName, const QString model, const QString pluginType,
                             const bool isUsed);
  void notifyBackClicked();

 protected slots:
  void onItemClicked(QListWidgetItem* item);
};
