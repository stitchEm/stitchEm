// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "ui_newaudioprocessorwidget.h"

class QLabel;
class NewAudioProcessorWidget : public QWidget, public Ui::NewAudioProcessorWidgetClass {
  Q_OBJECT

 public:
  explicit NewAudioProcessorWidget(QWidget* const parent = nullptr);

  void insertProcessorItem(const QString displayName, const QString type, const bool isUsed);
  void clearDevices();

 signals:
  void notifyProcessorSelected(const QString displayName, const QString type, const bool isUsed);
  void notifyBackClicked();

 protected slots:
  void onItemClicked(QListWidgetItem* item);
};
