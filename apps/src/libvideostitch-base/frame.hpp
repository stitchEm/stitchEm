// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include <QMutex>
#include <QReadWriteLock>

#include <cstdint>

/**
 * High precision date or time interval
 *
 * Store a high precision date or time interval. The maximum precision is the
 * microsecond, and a 64 bits integer is used to avoid overflows (maximum
 * time interval is then 292271 years, which should be long enough for any
 * video). Dates are stored as microseconds since a common date (eg. the epoch for
 * the system clock). Note that date and time intervals can be manipulated using
 * regular arithmetic operators, and that no special functions are required.
 */
typedef int64_t mtime_t;

/**
 * @brief The Frame class
 *
 * It is a Qt-friendly frame buffer, that can be passed through
 * a shared pointer between video players callbacks (output writers)
 * and widgets.
 *
 * Aside from that, just a lockable wrapper around a pixel buffer.
 */
class VS_COMMON_EXPORT Frame {
 public:
  Frame(char* buffer, unsigned w, unsigned h) : buf(buffer), date(0), width(w), height(h) {}
  Frame(unsigned w, unsigned h) : buf(nullptr), date(0), width(w), height(h) {}
  virtual ~Frame() {}

  char* buffer() { return buf; }

  void setDate(mtime_t d) { date = d; }
  mtime_t getDate() const { return date; }

  unsigned getWidth() const { return width; }
  unsigned getHeight() const { return height; }

  virtual void readLock() = 0;
  virtual void writeLock() = 0;
  virtual void unlock() = 0;

 protected:
  char* buf;
  mtime_t date;
  unsigned width;
  unsigned height;

  Frame(const Frame& o) {
    buf = o.buf;
    width = o.width;
    height = o.height;
    date = o.date;
  }
  Frame& operator=(const Frame&);
};

class VS_COMMON_EXPORT PanoFrame : public Frame {
 public:
  PanoFrame(unsigned w, unsigned h) : Frame(new char[w * h * 4], w, h) {}
  virtual ~PanoFrame() { delete[] buf; }

  virtual void readLock() { mutex.lock(); }
  virtual void writeLock() { mutex.lock(); }
  virtual void unlock() { mutex.unlock(); }

 private:
  QMutex mutex;

  PanoFrame(const PanoFrame&);
  PanoFrame& operator=(const PanoFrame&);
};

class VS_COMMON_EXPORT EyeFrame : public Frame {
 public:
  EyeFrame(char* buf, QReadWriteLock& lk, unsigned w, unsigned h) : Frame(buf, w, h), rwLk(lk) {}
  EyeFrame(const EyeFrame& o) : Frame(o), rwLk(o.rwLk) {}

  virtual void readLock() { rwLk.lockForRead(); }
  virtual void writeLock() { rwLk.lockForWrite(); }
  virtual void unlock() { rwLk.unlock(); }

 private:
  QReadWriteLock& rwLk;

  EyeFrame& operator=(const EyeFrame&);
};

class VS_COMMON_EXPORT StereoFrame : public Frame {
 public:
  StereoFrame(unsigned w, unsigned h)
      : Frame(new char[w * h * 4], w, h),
        leftFrame(buf, rwLk, width, height / 2),
        rightFrame(buf + width * height * 2, rwLk, width, height / 2) {}
  ~StereoFrame() { delete[] buf; }

  EyeFrame& getLeftFrame() { return leftFrame; }
  EyeFrame& getRightFrame() { return rightFrame; }

  virtual void readLock() { rwLk.lockForRead(); }
  virtual void writeLock() { rwLk.lockForWrite(); }
  virtual void unlock() { rwLk.unlock(); }

 private:
  EyeFrame leftFrame;
  EyeFrame rightFrame;
  QReadWriteLock rwLk;

  StereoFrame(const StereoFrame&);
  StereoFrame& operator=(const StereoFrame&);
};
