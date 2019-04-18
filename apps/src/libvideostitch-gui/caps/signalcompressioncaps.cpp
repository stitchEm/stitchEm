// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "signalcompressioncaps.hpp"

SignalCompressionCaps* SignalCompressionCaps::create() { return new SignalCompressionCaps(); }

std::shared_ptr<SignalCompressionCaps> SignalCompressionCaps::createOwned() {
  return std::shared_ptr<SignalCompressionCaps>(create(),
                                                [](SignalCompressionCaps* compressor) { compressor->autoDelete(); });
}

SignalCompressionCaps::SignalCompressionCaps() : inFlight(0), autoDeleteOn(false) {}

SignalCompressionCaps::~SignalCompressionCaps() {
  Q_ASSERT(autoDeleteOn);
  Q_ASSERT(inFlight == 0);
}

SignalCompressionCaps* SignalCompressionCaps::add() {
  std::lock_guard<std::mutex> lock(mutex);
  Q_ASSERT(inFlight >= 0);
  ++inFlight;
  return this;
}

int SignalCompressionCaps::nb() const { return inFlight; }

void SignalCompressionCaps::autoDelete() {
  // NOT QMutexLocker locking since *this may not exist anymore when returning.
  mutex.lock();
  Q_ASSERT(inFlight >= 0);
  autoDeleteOn = true;
  // The counter may already be 0
  if (inFlight == 0) {
    mutex.unlock();
    delete this;  // Hara-kiri.
  } else {
    mutex.unlock();
  }
}

int SignalCompressionCaps::pop() {
  // NOT QMutexLocker locking since *this may not exist anymore when returning.
  mutex.lock();
  Q_ASSERT(inFlight > 0);
  --inFlight;
  int returnValue = inFlight;
  if (autoDeleteOn && inFlight == 0) {
    mutex.unlock();
    delete this;  // Hara-kiri.
  } else {
    mutex.unlock();
  }
  return returnValue;
}
