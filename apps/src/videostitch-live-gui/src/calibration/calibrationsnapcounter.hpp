// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QFrame>

class QLabel;

class CalibrationSnapCounter : public QFrame {
  Q_OBJECT
 public:
  explicit CalibrationSnapCounter(QWidget* const parent = nullptr);

  void startCounter(const int value);

 signals:
  void notifyTimerEnded();

 private slots:
  void onTimerUpdate();

 private:
  QTimer* timer;
  QLabel* labelCounterValue;
  int counterValue;
  int ticNumber;
};
