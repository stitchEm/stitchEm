// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/utils/semaphore.hpp"

namespace VideoStitch {

/**
 * Read/Write mutex.
 */
class RWMutex {
 public:
  virtual ~RWMutex();
  virtual void writerLock() = 0;
  virtual void writerUnlock() = 0;
  virtual void readerLock() = 0;
  virtual void readerUnlock() = 0;
};

/**
 * Read RAII on Read/Write mutex.
 */
class ScopedReaderLock {
 public:
  explicit ScopedReaderLock(RWMutex& rwMutex) : rwMutex(rwMutex) { rwMutex.readerLock(); }
  ~ScopedReaderLock() { rwMutex.readerUnlock(); }

 private:
  RWMutex& rwMutex;
};

/**
 * Write RAII on Read/Write mutex.
 */
class ScopedWriterLock {
 public:
  explicit ScopedWriterLock(RWMutex& rwMutex) : rwMutex(rwMutex) { rwMutex.writerLock(); }
  ~ScopedWriterLock() { rwMutex.writerUnlock(); }

 private:
  RWMutex& rwMutex;
};

/**
 * A RW mutex that gives readers the priority. Will starve the writer.
 */
class ReaderPriorityMutex : public RWMutex {
 public:
  ReaderPriorityMutex();
  void writerLock();
  void writerUnlock();
  void readerLock();
  void readerUnlock();

 private:
  Semaphore writerSem;
  Semaphore readerSem;
  int pendingReads;
};

}  // namespace VideoStitch
