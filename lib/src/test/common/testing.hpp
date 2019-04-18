// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/imageOps.hpp"

#include <cuda/error.hpp>
#include "libvideostitch/config.hpp"

#include <csignal>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <sstream>
#include <math.h>
#if defined(_MSC_VER)
#include <Windows.h>
#define sleep(t) Sleep(1000 * (t))
#define backtrace(a, b) 0
#define backtrace_symbols_fd(a, b, c)
#else
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

void initTest() { std::signal(SIGABRT, handler); }

void ENSURE(bool condition, const char* msg = "") {
  if (!condition) {
    std::cerr << "TEST FAILED: " << msg << std::endl;
    std::raise(SIGABRT);
  }
}

void ENSURE(Status status, const char* msg = "") {
  if (!status.ok()) {
    std::cerr << "TEST FAILED: " << msg << std::endl;
    std::raise(SIGABRT);
  }
}

void ENSURE_CUDA(cudaError success, const char* msg = "") {
  if (success != cudaSuccess) {
    std::cerr << "TEST FAILED: " << msg << std::endl;
    std::raise(SIGABRT);
  }
}

void DIE(const char* msg) { ENSURE(false, msg); }

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
void ENSURE_NEQ(const T& a, const T& b) {
  if (!(a != b)) {
    std::cerr << "TEST FAILED: Expected not equal to '" << a << "', got '" << b << "'" << std::endl;
    std::raise(SIGABRT);
  }
}

template <typename T>
void ENSURE_APPROX_EQ(const T& a, const T& b, const T& eps) {
  T err = (T)fabs((double)a - (double)b);
  if (err > eps) {
    std::cerr << "TEST FAILED: Expected '" << a << "', got '" << b << "' (error=" << err << ">eps=" << eps << ")"
              << std::endl;
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

void ENSURE_RGBA210_EQ(uint32_t a, uint32_t b) {
  ENSURE_EQ(Image::RGB210::a(a), Image::RGB210::a(b));
  if (Image::RGB210::a(a)) {
    ENSURE_EQ(Image::RGB210::r(a), Image::RGB210::r(b));
    ENSURE_EQ(Image::RGB210::g(a), Image::RGB210::g(b));
    ENSURE_EQ(Image::RGB210::b(a), Image::RGB210::b(b));
  }
}

void ENSURE_RGBA210_ARRAY_EQ(const uint32_t* exp, const uint32_t* actual, std::size_t w, std::size_t h) {
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const uint32_t& expValue = exp[y * w + x];
      const uint32_t& actualValue = actual[y * w + x];
      if (Image::RGB210::a(expValue) != Image::RGB210::a(actualValue)) {
        std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected alpha=" << Image::RGB210::a(expValue)
                  << ", got alpha=" << Image::RGB210::a(actualValue) << std::endl;
        ENSURE_EQ(Image::RGB210::a(expValue), Image::RGB210::a(actualValue));
      } else if (Image::RGB210::a(expValue)) {
        if (!(expValue == actualValue)) {
          std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected '(" << Image::RGB210::r(expValue)
                    << "," << Image::RGB210::g(expValue) << "," << Image::RGB210::b(expValue) << ")', got '("
                    << Image::RGB210::r(actualValue) << "," << Image::RGB210::g(actualValue) << ","
                    << Image::RGB210::b(actualValue) << ")'" << std::endl;
          ENSURE_RGBA210_EQ(expValue, actualValue);
        }
      }
    }
  }
}

void ENSURE_RGBA8888_ARRAY_EQ(const uint32_t* exp, const uint32_t* actual, std::size_t w, std::size_t h) {
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const uint32_t& expValue = exp[y * w + x];
      const uint32_t& actualValue = actual[y * w + x];
      if (Image::RGBA::a(expValue) == 0xff) {
        if (!(expValue == actualValue)) {
          std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected '(" << Image::RGBA::r(expValue)
                    << "," << Image::RGBA::g(expValue) << "," << Image::RGBA::b(expValue) << ")', got '("
                    << Image::RGBA::r(actualValue) << "," << Image::RGBA::g(actualValue) << ","
                    << Image::RGBA::b(actualValue) << ")'" << std::endl;
          ENSURE_EQ(expValue, actualValue);
        }
      }
      ENSURE_EQ(Image::RGBA::a(expValue), Image::RGBA::a(actualValue));
    }
  }
}

#ifdef __CUDACC__

#define DEVICE_RUN(kernelFun)                                       \
  {                                                                 \
    dim3 oneOne(1, 1, 1);                                           \
    kernelFun<<<oneOne, oneOne>>>();                                \
    ::VideoStitch::Cuda::debugCudaErrors(cudaStreamSynchronize(0)); \
  }

__device__ void DEVICE_ENSURE_EQ(int a, int b) {
  if (a != b) {
    //     printf("TEST FAILED: Expected '%i', got '%i'\n", a, b);
    assert(false);
  }
}

__device__ void DEVICE_ENSURE_APPROX_EQ(float a, float b, float epsilon) {
  if (fabs(a - b) > epsilon) {
    //     printf("TEST FAILED: Expected '%f', got '%f'\n", a, b);
    assert(false);
  }
}
#endif  // __CUDACC__

}  // namespace Testing
}  // namespace VideoStitch
