// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SIGNALHANDLER_HPP
#define SIGNALHANDLER_HPP

#include "libvideostitch-gui/mainwindow/uniqueqapplication.hpp"

#include <stdexcept>
#include <signal.h>
#include <iostream>
#include <array>

static const std::array<int, 6> handledSignals = {{SIGINT, SIGTERM, SIGILL, SIGFPE, SIGABRT, SIGSEGV}};

/**
 * @brief The SignalHandler class
 */
class SignalHandler {
 public:
  /**
   * @brief Handles the application exit
   */
  static void handleExit(int) {
#ifdef Q_OS_UNIX
    // Clean up once, don't handle any signals after that (go back to default)
    struct sigaction signal_action;
    signal_action.sa_handler = SIG_DFL;
    for (auto signal : handledSignals) {
      sigaction(signal, &signal_action, NULL);
    }
#endif  // Q_OS_UNIX

    UniqueQApplication* application = qobject_cast<UniqueQApplication*>(qApp);
    if (application) {
      application->cleanup();
    }
  }

  /**
   * @brief Sets handleExit as signal handler for all the main signals
   */
  static void setupHandlers() {
#ifdef Q_OS_UNIX
    struct sigaction signal_action;
    signal_action.sa_handler = handleExit;

    // While handling one signal, block all signals we registered to handle
    sigset_t block_mask;
    sigemptyset(&block_mask);
    for (auto signal : handledSignals) {
      sigaddset(&block_mask, signal);
    }
    signal_action.sa_mask = block_mask;

    for (auto signal : handledSignals) {
      sigaction(signal, &signal_action, NULL);
    }

#else  // Q_OS_UNIX
    signal((int)SIGINT, SignalHandler::handleExit);
    signal((int)SIGTERM, SignalHandler::handleExit);
    signal((int)SIGILL, SignalHandler::handleExit);
    signal((int)SIGFPE, SignalHandler::handleExit);
    signal((int)SIGABRT, SignalHandler::handleExit);
    signal((int)SIGSEGV, SignalHandler::handleExit);
#endif
  }
};

#endif  // SIGNALHANDLER_HPP
