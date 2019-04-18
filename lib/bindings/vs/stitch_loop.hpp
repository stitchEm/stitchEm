// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../../include/libvideostitch/logging.hpp"
#include "../../include/libvideostitch/controller.hpp"
#include "../../include/libvideostitch/stitchOutput.hpp"

#ifdef __linux__
#include <GLFW/glfw3.h>
#else
#include <glfw/glfw3.h>
#endif
#include <chrono>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <sys/time.h>

static std::string LOOPtag = "StitchLoop";

/** @brief Stitching loop

*/
class StitchLoop {
 public:
  StitchLoop(GLFWwindow* _window, VideoStitch::Core::PotentialController& _controller,
             VideoStitch::Core::StitchOutput* _output)
      : window(_window),
        controller(_controller),
        output(_output),
        state(idle),
        stitchStatus(VideoStitch::Core::ControllerStatus::OK()),
        statusCached(false),
        algo(nullptr),
        stitchLoop(&StitchLoop::loop, this) {}

  ~StitchLoop() { Stop(); }

  VideoStitch::Status StitchStatus() {
    if (!statusCached) {
      // Cache the status to avoid having to wait on the status
      std::lock_guard<std::mutex> lock(stitchMutex);
      stitchStatusCache = stitchStatus;
      statusCached = true;
    }

    if (stitchStatusCache.getCode() == VideoStitch::Core::ControllerStatusCode::EndOfStream) {
      return {Origin::Input, ErrType::RuntimeError, "Could not load input frames, reader reported end of stream"};
    }

    return stitchStatusCache.getStatus();
  }

  void Start() { setState(starting, stitching); }

  void Stop() {
    if (setState(stopping)) {
      stitchLoop.join();
    }
  }

  void Unlock() { setState(stitching); }

  void Lock() { setState(idle); }

  void setAlgorithm(VideoStitch::Core::AlgorithmOutput* _algo,
                    const std::vector<VideoStitch::Core::ExtractOutput*>& _extracts) {
    if (_algo != algo) {
      std::lock_guard<std::mutex> _(stitchMutex);
      algo = _algo;
      extracts = _extracts;
    }
  }

 private:
  enum State { none, idle, starting, stitching, stopping };

  /** @brief Changes internal automata state, eventually waiting for it to reach a final sate.
   *  @return true if state has changed
   */
  bool setState(State newState, State finalState = none) {
    std::lock_guard<std::mutex> _(setStateMutex);

    std::unique_lock<std::mutex> lock(stateMutex);

    if (state == stopping) {
      return false;
    }

    if (state != newState) {
      state = newState;
    } else {
      return false;
    }

    if (finalState != none) {
      stateCV.wait(lock, [this, finalState] { return state == finalState; });
    }
    return true;
  }

  void changeState(State newState) {
    state = newState;
    stateCV.notify_one();
    stateMutex.unlock();
  }

  void loop() {
    bool stop = false;

    while (!stop) {
      stateMutex.lock();
      switch (state) {
        case starting:
          glfwMakeContextCurrent(window);

        case stitching: {
          std::lock_guard<std::mutex> _(stitchMutex);
          stitchStatus = controller->stitchAndExtract(output, extracts, algo, true);
          statusCached = stitchStatus.getCode() == stitchStatusCache.getCode();
        }
          changeState(stitching);
          std::this_thread::yield();
          break;

        case idle:
          changeState(idle);
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          break;

        case stopping:
          glfwMakeContextCurrent(nullptr);
          changeState(stopping);
          stop = true;
          break;
      }
    }
  }

  VideoStitch::Core::PotentialController& controller;
  VideoStitch::Core::StitchOutput* output;
  std::vector<VideoStitch::Core::ExtractOutput*> extracts;
  VideoStitch::Core::AlgorithmOutput* algo;

  GLFWwindow* window;

  State state;
  std::mutex setStateMutex;
  std::mutex stateMutex;
  std::condition_variable stateCV;
  std::mutex stitchMutex;
  std::thread stitchLoop;

  VideoStitch::Core::ControllerStatus stitchStatus;
  VideoStitch::Core::ControllerStatus stitchStatusCache;
  std::atomic<bool> statusCached;
};
