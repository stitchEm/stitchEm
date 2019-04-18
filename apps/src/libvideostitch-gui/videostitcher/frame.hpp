// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-base/frame.hpp"

#include "libvideostitch/frame.hpp"

#include <QMutex>

#include <condition_variable>
#include <memory>
#include <mutex>

class DevicePanoFrame : public Frame {
 public:
  DevicePanoFrame(unsigned w, unsigned h, VideoStitch::PixelFormat format)
      : Frame(w, h), textureLock(nullptr), ready(false), format(format) {}
  virtual ~DevicePanoFrame() {}

  void initialize(std::shared_ptr<std::mutex> mu) {
    {
      std::lock_guard<std::mutex> lk(m);
      textureLock = mu;
      ready = true;
    }
    cv.notify_one();
  }

  bool isReady() {
    std::lock_guard<std::mutex> lk(m);
    return ready;
  }

  void readLock() {
    waitReady();
    textureLock.get()->lock();
  }
  void writeLock() {
    waitReady();
    textureLock.get()->lock();
  }
  void unlock() {
    waitReady();
    textureLock.get()->unlock();
  }

  VideoStitch::PixelFormat getFormat() { return format; }

 private:
  void waitReady() {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return ready; });
  }

  std::shared_ptr<std::mutex> textureLock;
  std::mutex m;
  std::condition_variable cv;
  bool ready;
  VideoStitch::PixelFormat format;

  DevicePanoFrame(const DevicePanoFrame&);
  DevicePanoFrame& operator=(const DevicePanoFrame&);
};
