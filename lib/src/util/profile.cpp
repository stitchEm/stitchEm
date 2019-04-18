// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>
#include <sstream>

namespace VideoStitch {
namespace Util {
SimpleProfiler::SimpleProfiler(const char* title, bool usecs, ThreadSafeOstream& out)
    : _out(out), _title(title), usecs(usecs) {
  resetTimer();
}

SimpleProfiler::~SimpleProfiler() {
  const uint64_t duration = computeDuration();
  std::stringstream msg;
  msg << _title << ": ";
  if (usecs) {
    msg << duration << " usec" << std::endl;
  } else {
    msg << duration / 1000 << " ms" << std::endl;
  }
  _out << msg.str();
}

void SimpleProfiler::resetTimer() {
#ifdef _MSC_VER
  QueryPerformanceCounter(&_tv);
#else
  gettimeofday(&_tv, NULL);
#endif
}

uint64_t SimpleProfiler::computeDuration() {
#ifdef _MSC_VER
  LARGE_INTEGER stop;
  QueryPerformanceCounter(&stop);
  LARGE_INTEGER countsPerSecond;
  QueryPerformanceFrequency(&countsPerSecond);
  uint64_t duration = (uint64_t)((1000000 * (stop.QuadPart - _tv.QuadPart)) / countsPerSecond.QuadPart);
#else
  struct timeval stop;
  gettimeofday(&stop, NULL);
  uint64_t duration = ((uint64_t)stop.tv_sec * 1000000 + stop.tv_usec) - ((uint64_t)_tv.tv_sec * 1000000 + _tv.tv_usec);
#endif
  return duration;
}

SimpleTimer::SimpleTimer() : SimpleProfiler("", true, Logger::get(Logger::Info)) {  // dummy parameters
}

void SimpleTimer::reset() { return resetTimer(); }

uint64_t SimpleTimer::elapsed() { return computeDuration(); }

}  // namespace Util
}  // namespace VideoStitch
