// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#if defined(__GNUC__)
#define VS_PLUGINS_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define VS_PLUGINS_EXPORT __declspec(dllexport)
#else
#error
#endif
