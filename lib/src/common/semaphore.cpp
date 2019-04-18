// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/utils/semaphore.hpp"

namespace VideoStitch {

Semaphore::Semaphore(int initialCount) : count(initialCount) {}

void Semaphore::notify() {
  std::lock_guard<std::mutex> lock(mutex);
  ++count;
  condition.notify_one();
}

void Semaphore::wait() {
  std::unique_lock<std::mutex> lock(mutex);
  while (count == 0) {
    condition.wait(lock);
  }
  --count;
}

bool Semaphore::wait_for(unsigned timeOutMs) {
  std::chrono::milliseconds timeDur(timeOutMs);
  std::unique_lock<std::mutex> lock(mutex);
  while (count == 0) {
    if (condition.wait_for(lock, timeDur) == std::cv_status::timeout) {
      return false;
    }
  }
  --count;
  return true;
}
}  // namespace VideoStitch
