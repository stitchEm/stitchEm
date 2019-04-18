// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "pretty_print.hpp"

#include <csignal>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <math.h>
#include <sstream>
#include <thread>
#if defined(_MSC_VER)
#include <Windows.h>
#define sleep(t) Sleep(1000 * (t))
#define backtrace(a, b) 0
#define backtrace_symbols_fd(a, b, c) (void*)&b
#else
#include <unistd.h>
#if !defined(__ANDROID__)
#include <execinfo.h>
#else
#include <cstdlib>
#define backtrace(a, b) 0
#define backtrace_symbols_fd(a, b, c) (void*)&b
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-value"
#endif /*(!defined __ANDROID__)*/
#endif

namespace VideoStitch {
namespace Testing {

#ifdef _MSC_VER
// MSVC is dumb here
#pragma warning(push)
#pragma warning(disable : 4101 4189)
#endif

void handler(int sig) {
  void* array[100];
  const int size = backtrace(array, 100);
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, 2);
  exit(1);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

void initTest() {
// GCC doesn't like '&&'?
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  // disable signal handling / backtrace printing when running ThreadSanitizer
#else
  std::signal(SIGABRT, handler);
#endif
#else
  std::signal(SIGABRT, handler);
#endif
}

void ENSURE(bool condition, const char* msg = "") {
  if (!condition) {
    std::cerr << "TEST FAILED: " << msg << std::endl;
    std::raise(SIGABRT);
  }
}

void DIE(const char* msg) { ENSURE(false, msg); }

void initTestWithTimeoutInSeconds(int seconds) {
  initTest();
  std::thread timeOutThread{[seconds]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(seconds * 1000));
    DIE("Test timed out");
  }};
  timeOutThread.detach();
}

template <typename T>
void ENSURE_EQ(const T& a, const T& b, const char* msg = "") {
  if (!(a == b)) {
    std::stringstream ss;
    ss << "TEST FAILED: Expected '" << a << "', got '" << b << "' " << msg << std::endl;
    std::cerr << ss.str();
    std::raise(SIGABRT);
  }
}

template <typename T>
void ENSURE_NEQ(const T& a, const T& b, const char* msg = "") {
  if (!(a != b)) {
    std::cerr << "TEST FAILED: Expected not equal to '" << a << "', got '" << b << "'" << msg << std::endl;
    std::raise(SIGABRT);
  }
}

template <typename T>
void ENSURE_APPROX_EQ(const T& a, const T& b, const T& eps, const char* msg = "") {
  T err = (T)fabs((double)a - (double)b);
  if (err > eps) {
    std::cerr << "TEST FAILED: Expected '" << a << "', got '" << b << "' (error=" << err << ">eps=" << eps << ") "
              << msg << std::endl;
    std::raise(SIGABRT);
  }
}

template <typename T>
void ENSURE_ARRAY_EQ(const T* exp, const T* actual, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    if (!(exp[i] == actual[i])) {
      std::cerr << "TEST FAILED: At index '" << i << "', expected '" << exp[i] << "', got '" << actual[i] << "'"
                << std::endl;
      std::raise(SIGABRT);
    }
  }
}

template <typename T>
void ENSURE_2D_ARRAY_EQ(const T* exp, const T* actual, std::size_t w, std::size_t h) {
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const T& expValue = exp[y * w + x];
      const T& actualValue = actual[y * w + x];
      if (!(expValue == actualValue)) {
        std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected '" << expValue << "', got '"
                  << actualValue << "'" << std::endl;
        std::raise(SIGABRT);
      }
    }
  }
}

template <typename T>
void ENSURE_ARRAY_NEQ(const T* exp, const T* actual, std::size_t size) {
  bool eq = true;
  for (std::size_t i = 0; i < size; ++i) {
    eq = eq && (exp[i] == actual[i]);
  }
  ENSURE(!eq, "Expected arrays to differ, but they are the same.");
}

std::string getDataFolder() {
  char const* testData = std::getenv("VS_TEST_DATA_DIR");
  ENSURE((testData != NULL), "VS_TEST_DATA_DIR environment variable should be set");
  return std::string(testData);
}

}  // namespace Testing
}  // namespace VideoStitch
