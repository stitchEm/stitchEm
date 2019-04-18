// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROFILE_HPP_
#define PROFILE_HPP_

#include "config.hpp"

#include <iosfwd>

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <sys/time.h>
#endif

// macro-based profiling, enabled only when -DSIMPLEPROFILEON
#ifdef SIMPLEPROFILEON
#define SIMPLEPROFILE_MS(title) Util::SimpleProfiler prof((title), false, Logger::get(Logger::Debug))
#else
#define SIMPLEPROFILE_MS(title) (void)0
#endif

// GPU/CPU timeline analysis, enabled only with -DUSE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"
#else
typedef int nvtxRangeId_t;
#define nvtxRangeStartA(a) 0
#define nvtxRangeEnd(a) ((a)++)
#define nvtxMarkA(a)
#endif

namespace VideoStitch {

class ThreadSafeOstream;

namespace Util {
/**
 * @brief A simple scoped time-based profiler.
 */
class VS_EXPORT SimpleProfiler {
 public:
  /**
   * Create a profiler.
   * @param title Message to display.
   * @param usecs If true, display microseconds. Else, display milliseconds.
   * @param out Where to write the message.
   */
  SimpleProfiler(const char* title, bool usecs, ThreadSafeOstream& out);

  ~SimpleProfiler();

 protected:
  /**
   * Get the initial time.
   */
  void resetTimer();
  /**
   * Compute duration between the initial time and now.
   */
  uint64_t computeDuration();

 private:
#ifdef _MSC_VER
  LARGE_INTEGER _tv;
#else
  struct timeval _tv;
#endif
  ThreadSafeOstream& _out;
  const char* const _title;
  bool usecs;
};

/**
 * @brief A simple timer.
 */
class VS_EXPORT SimpleTimer : protected SimpleProfiler {
 public:
  SimpleTimer();

  /**
   * Reset the timer.
   */
  void reset();

  /**
   * Return the elapsed time (in microseconds) since the time was created (or reset).
   */
  uint64_t elapsed();
};

}  // namespace Util
}  // namespace VideoStitch

#endif
