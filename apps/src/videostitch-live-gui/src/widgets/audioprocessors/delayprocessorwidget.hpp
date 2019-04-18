// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include "audioprocessorconfigurationwidget.hpp"

namespace Ui {
class DelayProcessorWidget;
}

class LiveAudioProcessorDelay;

class DelayProcessorWidget : public AudioProcessorConfigurationWidget {
  Q_OBJECT

 public:
  explicit DelayProcessorWidget(LiveAudioProcessorDelay *ref, QWidget *parent = nullptr);
  ~DelayProcessorWidget();
  virtual LiveAudioProcessFactory *getConfiguration() const override;
  virtual void loadConfiguration() override;

 protected:
  virtual void reactToChangedProject() override;

 private slots:
  void onSliderChanged(const int value);

 private:
  LiveAudioProcessorDelay *delayRef;
  Ui::DelayProcessorWidget *ui;
};
