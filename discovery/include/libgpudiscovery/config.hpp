// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

// exported symbols
#if defined(__GNUC__)
#define VS_DISCOVERY_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#ifdef VS_LIB_DISCOVERY
#define VS_DISCOVERY_EXPORT __declspec(dllexport)
#else
#define VS_DISCOVERY_EXPORT __declspec(dllimport)
#endif
#endif
