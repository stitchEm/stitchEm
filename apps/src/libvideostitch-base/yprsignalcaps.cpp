// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "yprsignalcaps.hpp"

#include "geometry.hpp"

YPRSignalCaps* YPRSignalCaps::create() { return new YPRSignalCaps(); }

YPRSignalCaps::YPRSignalCaps() : yaw(0.0), pitch(0.0), roll(0.0), inFlight(0), term(false) {}

YPRSignalCaps::~YPRSignalCaps() {
  Q_ASSERT(term);
  Q_ASSERT(inFlight == 0);
}

YPRSignalCaps* YPRSignalCaps::add(double y, double p, double r) {
  std::lock_guard<std::mutex> lock(mutex);
  Q_ASSERT(inFlight >= 0);
  ++inFlight;
  qreal o0, o1, o2, o3;
  euler2quaternion(yaw, pitch, roll, o0, o1, o2, o3);
  qreal p0, p1, p2, p3;
  y = degToRad(y);
  p = degToRad(p);
  r = degToRad(r);
  euler2quaternion(y, p, r, p0, p1, p2, p3);
  qreal q0, q1, q2, q3;
  quaternionProduct(o0, o1, o2, o3, p0, p1, p2, p3, q0, q1, q2, q3);
  quaternion2euler(q0, q1, q2, q3, yaw, pitch, roll);
  return this;
}

void YPRSignalCaps::terminate() {
  // NOT QMutexLocker locking since *this may not exist anymore when returning.
  mutex.lock();
  term = true;
  // The counter may already be 0
  if (inFlight == 0) {
    mutex.unlock();
    delete this;  // Hara-kiri.
  } else {
    mutex.unlock();
  }
}

void YPRSignalCaps::popAll(double& y, double& p, double& r) {
  // NOT QMutexLocker locking since *this may not exist anymore when returning.
  mutex.lock();
  Q_ASSERT(inFlight > 0);
  --inFlight;
  y = radToDeg(yaw);
  p = radToDeg(pitch);
  r = radToDeg(roll);
  yaw = 0.0;
  pitch = 0.0;
  roll = 0.0;
  if (term && inFlight == 0) {
    mutex.unlock();
    delete this;  // Hara-kiri.
  } else {
    mutex.unlock();
  }
}
