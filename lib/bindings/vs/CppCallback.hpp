// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <functional>
#include <iostream>
#include <thread>

class CppCallback {
 public:
  virtual ~CppCallback() {}
  virtual void operator()(const std::string&) {
    std::cout << "\n Beware: I'm a base class.\n";
    assert(false);
  }

  virtual void join() {
    if (pythonCallbackThread.joinable()) {
      // as this method will always be called from python,
      // we release the python lock so the thread can acquire it
      // see pythonCallback method
      PyThreadState* _save;
      _save = PyEval_SaveThread();
      pythonCallbackThread.join();
      PyEval_RestoreThread(_save);
    }
  }

  virtual std::function<void(const std::string&)> toFunction() {
    return [this](const std::string& payload) {
      // this will be called from unknown places that can have acquired some locks
      // so we use a separate thread to avoid waiting for python lock while keeping these other locks
      // as this leads to deadlocks situations
      // if there was an unfinished previous call, it is moved to the new thread, where it will be waited for
      pythonCallbackThread = std::thread(&CppCallback::pythonCallback, this, payload, std::move(pythonCallbackThread));
    };
  }

  void pythonCallback(const std::string& payload, std::thread&& previousCall) {
    // wait for previous call to be completed
    if (previousCall.joinable()) {
      previousCall.join();
    }

    // As this will be called from a thread python knows nothing about,
    // we need to acquire GIL manually and only then we can execute python code.
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    (*this)(payload);
    PyGILState_Release(gstate);
  }

 private:
  std::thread pythonCallbackThread;
};
