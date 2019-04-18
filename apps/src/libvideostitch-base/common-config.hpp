// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef COMMONCONFIG_HPP
#define COMMONCONFIG_HPP

#if defined(__GNUC__)
#define VS_COMMON_EXPORT __attribute__((visibility("default")))
#define VS_COMMON_TEMPLATE_EXPORT
#elif defined(_MSC_VER)
#ifdef VS_LIB_COMMON_COMPILATION
#define VS_COMMON_EXPORT __declspec(dllexport)
#else
#define VS_COMMON_EXPORT __declspec(dllimport)
#endif
#define VS_COMMON_TEMPLATE_EXPORT VS_COMMON_EXPORT
#else
#error
#endif

#define VIDEOSTITCH_ORG_NAME "stitchEm"
#define VIDEOSTITCH_ORG_DOMAIN "https://github.com/stitchEm"
#define VIDEOSTITCH_STUDIO_APP_NAME "VideoStitch Studio"
#define VIDEOSTITCH_BATCH_STITCHER_APP_NAME "Batch Stitcher"
#define VAHANA_VR_APP_NAME "Vahana VR"
#define VIDEOSTITCH_STUDIO_SETTINGS_NAME "Studio"
#define VIDEOSTITCH_BATCH_STITCHER_SETTINGS_NAME "BatchStitcher"
#define VAHANA_VR_SETTINGS_NAME "VahanaVR"

#define VS_DEFAULT_INPUT "procedural:grid(size=30,color=%0)"

#define NVIDIA_WEBSITE "http://www.nvidia.com/"

#endif  // COMMONCONFIG_HPP
