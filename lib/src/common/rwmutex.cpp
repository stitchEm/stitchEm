// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rwmutex.hpp"

namespace VideoStitch {

RWMutex::~RWMutex() {}

// The implementations below are taken from http://en.wikipedia.org/wiki/Readers%E2%80%93writers_problem.

ReaderPriorityMutex::ReaderPriorityMutex() : writerSem(1), readerSem(1), pendingReads(0) {}

void ReaderPriorityMutex::writerLock() { writerSem.wait(); }

void ReaderPriorityMutex::writerUnlock() { writerSem.notify(); }

void ReaderPriorityMutex::readerLock() {
  readerSem.wait();
  ++pendingReads;
  if (pendingReads == 1) {
    writerSem.wait();
  }
  readerSem.notify();
}

void ReaderPriorityMutex::readerUnlock() {
  readerSem.wait();
  --pendingReads;
  if (pendingReads == 0) {
    writerSem.notify();
  }
  readerSem.notify();
}

}  // namespace VideoStitch
