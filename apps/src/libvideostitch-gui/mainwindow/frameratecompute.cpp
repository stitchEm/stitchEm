// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "frameratecompute.hpp"

#include <cmath>

#define MIN_SAMPLES 150
#define NUM_SAMPLES 1000

FramerateCompute::FramerateCompute() : QObject(), currentFramerate(0) {
  connect(&refreshTimer, &QTimer::timeout, this, &FramerateCompute::actualizeFramerate);
}

void FramerateCompute::start() {
  fpsTimer.start();
  refreshTimer.start(1000);
}

void FramerateCompute::stop() {
  currentFramerate = 0;
  frametimes.clear();
  refreshTimer.stop();
}

void FramerateCompute::restart() {
  stop();
  start();
}

float FramerateCompute::getFramerate() {
  QMutexLocker lock(&mu);
  return currentFramerate;
}

void FramerateCompute::actualizeFramerate() {
  QMutexLocker lock(&mu);
  if (frametimes.size() < MIN_SAMPLES) {
    currentFramerate = 0;
  } else {
    float sum = std::accumulate(frametimes.begin(), frametimes.end(), 0.0);
    currentFramerate = 1000.f * frametimes.size() / sum;
    currentFramerate = roundf(currentFramerate * 10) / 10;
  }
}

void FramerateCompute::tick() {
  QMutexLocker lock(&mu);
  frametimes.push_front(fpsTimer.elapsed());
  if (frametimes.size() > NUM_SAMPLES) {
    frametimes.pop_back();
  }
  fpsTimer.start();
}
