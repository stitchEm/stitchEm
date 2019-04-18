// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SIGNALCOMPRESSIONCAPS_HPP
#define SIGNALCOMPRESSIONCAPS_HPP

#include <memory>
#include <mutex>
#include "../common.hpp"

/**
 * @brief A class to implement signal compression.
 * To be used when an emitter can create a lot of signals that are heavyweight to process by the worker,
 * and where each signal cancels the effect of the previous one.
 * In that case, when the worker receives a signal, it may want to skip the front signal if there are similar enqueued
 * signals after it.
 *
 * Usage:
 * Emitter:
 *  1 - The emitter creates a SignalCompressionCaps object sigCompCap on the heap on construction.
 *  2 - The emitter emits mySignal(sigCompCap->addCount(), ...)
 *  3 - (as many times as wanted...)
 *  4 - When done, the emitter cannot delete sigCompCap, because there may remain unprocesses signals that reference it,
 * so it should call sigCompCap->autoDelete(). sigCompCap must of course not be touched afterwards.
 *
 * Worker slot:
 *  void mySlot(SignalCompressionCaps* comp, ...) {
 *    if (comp->pop() > 0) {
 *      // There are remaining mesages.
 *      return;
 *    }
 *    // Process the message. You are not allowed to touch comp.
 *  }
 *
 */
class VS_GUI_EXPORT SignalCompressionCaps {
 public:
  /**
   * Creates a compressor on the heap.
   */
  static SignalCompressionCaps* create();
  static std::shared_ptr<SignalCompressionCaps> createOwned();

  /**
   * Number of signals emitted.
   */
  int nb() const;

  /**
   * Adds a signal.
   * Called only by the emitter.
   */
  SignalCompressionCaps* add();

  /**
   * Called only by the emitter. Once this has been called, the emitter is not allowed touch the object anymore.
   */
  void autoDelete();

  /**
   * Removes a signal. If autoDelete() has been called, delete ourselves.
   * Called only by the worker, which is only allowed to call it once.
   * @return the number of remaining signals.
   */
  int pop();

 private:
  SignalCompressionCaps();
  ~SignalCompressionCaps();

 private:
  std::mutex mutex;
  int inFlight;
  bool autoDeleteOn;
};

#endif  // QUEUEDOPCAPS_HPP
