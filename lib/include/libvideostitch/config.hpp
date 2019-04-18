// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/**
 * This file defines some macros and preprocessor directives used by VideoStitch.
 */

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

// exported symbols
#if defined(__GNUC__)
#define VS_EXPORT __attribute__((visibility("default")))
#define VS_TEMPLATE_EXPORT
#define EXPIMP_TEMPLATE
#elif defined(_MSC_VER)
#ifdef VS_LIB_COMPILATION
#define VS_EXPORT __declspec(dllexport)
#define EXPIMP_TEMPLATE
#else
#define VS_EXPORT __declspec(dllimport)
#define EXPIMP_TEMPLATE extern
#endif
#elif not defined(SWIG)
#error
#endif

// deprecation
#ifdef __GNUC__
#define VS_DEPRECATED(func) func __attribute__((deprecated))
#elif defined(_MSC_VER)
#define VS_DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement the macro VS_DEPRECATED")
#define VS_DEPRECATED(func) func
#endif

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
 * Ultra High precision date or time interval
 *
 * Store a high precision date or time interval. The maximum precision is the
 * 0.01 nanosecond, and a 64 bits integer is used to avoid overflows (maximum
 * time interval is then 1067 days, which should be long enough for any
 * video). Dates are stored as 0.01 nanoseconds since a common date (eg. the epoch for
 * the system clock). Note that date and time intervals can be manipulated using
 * regular arithmetic operators, and that no special functions are required.
 */
typedef int64_t cntime_t;
#define MTOCNTIME(ts) cntime_t(cntime_t(100000) * (ts))
#define CNTOMTIME(ts) mtime_t((ts) / 100000)

/**
 * Frame number inside the container.
 *
 * This concerns file-based videos.
 * Direct conversion from mtime_t timestamp is possible by consulting
 * the frame rate of the video stitcher, accessible through the
 * Controller API.
 */
typedef int32_t frameid_t;

/**
 * Identifier of an audio track.
 *
 * Consult audio.hpp for the different layouts accessible.
 */
typedef int64_t channel_t;

/**
 * Input identifier.
 */
typedef int32_t readerid_t;
typedef int32_t videoreaderid_t;
typedef int32_t audioreaderid_t;
typedef int32_t overlayreaderid_t;

/**
 * Group identifier.
 */
typedef int32_t groupid_t;

/**
 * Specific value of lastFrame that indicates an unbounded sequence.
 */
#define NO_LAST_FRAME (std::numeric_limits<frameid_t>::max() - 1)

// Max input number
#define MAX_VIDEO_INPUTS 32

// Max overlay number
#define MAX_OVERLAYS 32

// Use these as default output size if current is invalid
#define FALLBACK_OUTPUT_WIDTH 2048
#define FALLBACK_OUTPUT_HEIGHT 1024

// Smooth image merger (gradient, laplacian) defines
#define DEFAULT_BLENDING_FEATHER 100

// Laplacian merger defaults
#define DEFAULT_BASE_LAPLACIAN_SIZE 64
#define DEFAULT_LAPLACIAN_GAUSSIAN_RADIUS 5
#define DEFAULT_LAPLACIAN_BLUR_PASSES 1
#define DEFAULT_GAUSSIAN_BLUR_RADIUS 3
// Gradient merger v2 defines
#define DEFAULT_MERGER_V2_FEATHERING 2.0f

// Laplacian merger v2 defines
#define DEFAULT_LAPLACIAN_MERGER_V2_FEATHERING 2.0f

// Default output frame rate
#define VIDEO_WRITER_DEFAULT_FRAMERATE_NUM 25
#define VIDEO_WRITER_DEFAULT_FRAMERATE_DEN 1

#undef _STDINT_H
#include <stdint.h>

#if defined(_MSC_VER)
#define strtoll _strtoi64
#endif

#if defined(_MSC_VER)
#define VS_ISNAN _isnan
#elif defined(__APPLE__) && !defined(__CUDACC__)
#define VS_ISNAN std::isnan
#else
#define VS_ISNAN std::isnan
#endif

#define MIN_FOV 0.1

// Initial/Default values for PTV.
#define PTV_DEFAULT_INPUTDEF_REDCB 1.0
#define PTV_DEFAULT_INPUTDEF_GREENCB 1.0
#define PTV_DEFAULT_INPUTDEF_BLUECB 1.0
#define PTV_DEFAULT_INPUTDEF_EXPOSURE 0.0
#define PTV_DEFAULT_INPUTDEF_EMORA 0.0
#define PTV_DEFAULT_INPUTDEF_EMORB 0.0
#define PTV_DEFAULT_INPUTDEF_EMORC 0.0
#define PTV_DEFAULT_INPUTDEF_EMORD 0.0
#define PTV_DEFAULT_INPUTDEF_EMORE 0.0
#define PTV_DEFAULT_INPUTDEF_GAMMA 1.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTA 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTB 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTC 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTC 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTP1 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTP2 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTS1 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTS2 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTS3 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTS4 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTTAU1 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDISTTAU2 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDIST_CENTER_X 0.0
#define PTV_DEFAULT_INPUTDEF_LENSDIST_CENTER_Y 0.0
#define PTV_DEFAULT_INPUTDEF_HORIZONTAL_FOCAL 1000.0
#define PTV_DEFAULT_INPUTDEF_VERTICAL_FOCAL 0.
#define PTV_DEFAULT_INPUTDEF_VIG_COEFF0 1.0
#define PTV_DEFAULT_INPUTDEF_VIG_COEFF1 0.0
#define PTV_DEFAULT_INPUTDEF_VIG_COEFF2 0.0
#define PTV_DEFAULT_INPUTDEF_VIG_COEFF3 0.0
#define PTV_DEFAULT_INPUTDEF_VIG_CENTER_X 0.0
#define PTV_DEFAULT_INPUTDEF_VIG_CENTER_Y 0.0
#define PTV_DEFAULT_INPUTDEF_PITCH 0.0
#define PTV_DEFAULT_INPUTDEF_ROLL 0.0
#define PTV_DEFAULT_INPUTDEF_YAW 0.0
#define PTV_DEFAULT_INPUTDEF_TRANS_X 0.0
#define PTV_DEFAULT_INPUTDEF_TRANS_Y 0.0
#define PTV_DEFAULT_INPUTDEF_TRANS_Z 0.0
#define PTV_DEFAULT_INPUTDEF_TEMPLATE_FOCAL_STD_DEV_VALUE_PERCENTAGE 5.0
#define PTV_DEFAULT_INPUTDEF_TEMPLATE_CENTER_STD_DEV_WIDTH_PERCENTAGE 10.0
#define PTV_DEFAULT_INPUTDEF_TEMPLATE_DISTORT_STD_DEV_VALUE_PERCENTAGE 50.0
#define PTV_DEFAULT_INPUTDEF_TEMPLATE_ANGLE_STD_DEV 5.0
#define PTV_DEFAULT_INPUTDEF_TEMPLATE_TRANSLATION_STD_DEV 0.0
#define PTV_DEFAULT_HFOV 120.0
#define PTV_DEFAULT_AUTO_FOV 0.0

#define PTV_DEFAULT_PANODEF_EXPOSURE 0.0
#define PTV_DEFAULT_PANODEF_REDCB 1.0
#define PTV_DEFAULT_PANODEF_GREENCB 1.0
#define PTV_DEFAULT_PANODEF_BLUECB 1.0
#define PTV_DEFAULT_PANODEF_STAB_YAW 0.0
#define PTV_DEFAULT_PANODEF_STAB_PITCH 0.0
#define PTV_DEFAULT_PANODEF_STAB_ROLL 0.0
#define PTV_DEFAULT_PANODEF_SPHERE_SCALE 1.0
#define PTV_DEFAULT_PANODEF_MIN_SPHERE_SCALE 0.01
#define PTV_DEFAULT_PANODEF_LENGTH 1024

#define PTV_DEFAULT_OVERLAY_SCALE 1.0
#define PTV_DEFAULT_OVERLAY_TRANSX 0.0
#define PTV_DEFAULT_OVERLAY_TRANSY 0.0
#define PTV_DEFAULT_OVERLAY_TRANSZ 0.0
#define PTV_DEFAULT_OVERLAY_ALPHA 1.0

enum Eye {
  LeftEye,
  RightEye,
};

enum AddressSpace { Device, Host };

#ifdef ANDROID__GNUSTL
#include <string>
#include <sstream>
#include <stdlib.h>
#include <stdexcept>

namespace std {
template <typename T>
string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

#if defined(__arm__)
template <typename T>
T round(T v) {
  return int(v + 0.5);
}
#endif

inline int stoi(const std::string& str, std::size_t* pos = 0, int base = 10) {
  const char* begin = str.c_str();
  char* end = nullptr;
  long value = strtol(begin, &end, base);

  if (errno == ERANGE || value > std::numeric_limits<int>::max()) {
    throw std::out_of_range("stoi: out of range");
  }

  if (end == str.c_str()) {
    throw std::invalid_argument("stoi: invalid argument");
  }

  if (pos) *pos = end - begin;

  return (int)value;
}

inline float stof(const string& str) { return (float)atof(str.c_str()); }
}  // namespace std
#endif
