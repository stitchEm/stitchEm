// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QObject>
#include <QTime>
#include <QTimer>
#include <deque>
#include <numeric>
#include <QMutex>

class VS_GUI_EXPORT FramerateCompute : public QObject {
  Q_OBJECT
 public:
  FramerateCompute();
  void start();
  void stop();
  void restart();
  void tick();
  float getFramerate();

 public slots:
  void actualizeFramerate();

 private:
  float currentFramerate;
  QTime fpsTimer;
  std::deque<quint32> frametimes;
  QTimer refreshTimer;
  QMutex mu;
};
