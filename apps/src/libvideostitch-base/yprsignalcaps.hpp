// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include <mutex>

/**
 * @brief A class to implement yaw/pitch/roll signal compression.
 * To be used when an emitter can create a lot of signals that are heavyweight to process by the worker,
 * where each signal add to the effet of the previous one,
 * and where the signals in-flight can be merged.
 * In that case, when the worker receives a signal, it may want to skip the front signal if there are similar enqueued
 * signal after it.
 *
 * Usage:
 * Emitter:
 *  1 - The emitter creates a YPRSignalCaps object sigCompCap on the heap on construction.
 *  2 - The emitter emits mySignal(sigCompCap->add(y, p, r))
 *  3 - (as many times as wanted...)
 *  4 - When done, the emitter cannot delete sigCompCap, because there may remain unprocesses signals that reference it,
 * so it should call sigCompCap->autoDelete(). sigCompCap must of course not be touched afterwards.
 *
 * Worker slot:
 *  void mySlot(SignalCompressionCaps* comp, ...) {
 *    // Pop all the signals in the queue
 *    double y, p, r;
 *    comp->popAll(y, p, r);
 *    // Process the message. You are not allowed to touch comp.
 *  }
 *
 */

class VS_COMMON_EXPORT YPRSignalCaps {
 public:
  /**
   * Creates a compressor on the heap.
   */
  static YPRSignalCaps* create();

  /**
   * Adds a signal.
   * Called only by the emitter.
   * Parameters are in degrees
   */
  YPRSignalCaps* add(double yaw, double pitch, double roll);

  /**
   * Removes all signals.
   * Called only by the worker, which is only allowed to call it once.
   * Parameters are in degrees
   */
  void popAll(double& yaw, double& pitch, double& roll);

  /**
   * Called only by the emitter. Once this has been called, the emitter is not allowed touch the object anymore.
   */
  void terminate();

 private:
  YPRSignalCaps();
  ~YPRSignalCaps();

  double yaw, pitch, roll;  //!< In radians
  std::mutex mutex;
  int inFlight;
  bool term;
};
