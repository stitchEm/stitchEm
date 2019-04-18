// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_audioprocessdetailwidget.h"

#include <QWidget>

class LiveAudioProcessFactory;

class AudioProcessDetailWidget : public QWidget {
  Q_OBJECT

 public:
  explicit AudioProcessDetailWidget(LiveAudioProcessFactory* processor, QWidget* parent = nullptr);
  ~AudioProcessDetailWidget();
  LiveAudioProcessFactory* getProcessor() const;

 signals:
  void notifyDeleteProcessor(const QString id);

 private slots:
  void onDeleteClicked();

 private:
  LiveAudioProcessFactory* processor;
  Ui::AudioProcessDetailWidgetClass* ui;
};
