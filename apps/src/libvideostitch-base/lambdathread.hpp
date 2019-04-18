// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include <QThread>

#include <functional>

typedef std::function<void()> Lambda;

class VS_COMMON_EXPORT LambdaThread : public QThread {
  Q_OBJECT
 public:
  template <typename F>
  LambdaThread(F f, bool runEventLoop = false) : f(f), runEventLoop(runEventLoop) {}

 private:
  Lambda f;
  bool runEventLoop;

  void run() {
    f();
    if (runEventLoop) {
      exec();
    }
  }
};
