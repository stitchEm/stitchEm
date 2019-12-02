// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef COMMON_HPP
#define COMMON_HPP

#include <qsystemdetection.h>

/**
 * README: this file is included automatically from the appsconfig.pri.
 */

/**
 * General parameters.
 */
#define URL_STYLE "<style type=text/css>a:link {color:orange; text-decoration:underline;}</style>"

// Tutorials
#define VIDEOSTITCH_YOUTUBE_STUDIO_URL "https://www.youtube.com/playlist?list=PLE5eSxUoYnqh8n6kjHKLZgIpbHONIIVtV"
#define VIDEOSTITCH_YOUTUBE_VAHANA_URL "https://www.youtube.com/playlist?list=PLE5eSxUoYnqgv4S8ayQ5_YsPP8x_MCvvM"
#define VIDEOSTITCH_URL "https://github.com/stitchEm"
#define VIDEOSTITCH_SUPPORT_URL "https://github.com/stitchEm"
#define VIDEOSTITCH_LIBRARIES_URL "https://github.com/stitchEm/stitchEm/blob/master/doc/LICENSE-3RD-PARTY-LIBRARIES.md"

// Color
#define ORAH_COLOR "FF9E00";

/**
 * System parameters.
 */
//#define VSNOTHREAD //All QObjects in the same thread. Useful for debugging purpose.
#ifndef VSNOTHREAD
#include <QLocale>
#include <QCoreApplication>
#define VS_TH_ASSERT() Q_ASSERT(QThread::currentThread() != QCoreApplication::instance()->thread());
#else
#define VS_TH_ASSERT()
#endif

#define VS_THREADSAFE_PTV  // PTV modifications are protected by a lock. Remove only to find lock issues.

/**
 * Safety checks.
 */
#ifdef NDEBUG
#if defined(VSNOTHREAD) || !defined(VS_THREADSAFE_PTV)
#error "Safety conditions no met. Please check your common.hpp file."
#endif
#endif

// exported symbols
#if defined(__GNUC__)
#define VS_GUI_EXPORT __attribute__((visibility("default")))
#define VS_GUI_TEMPLATE_EXPORT
#elif defined(_MSC_VER)
#ifdef VS_LIB_GUI_COMPILATION
#define VS_GUI_EXPORT __declspec(dllexport)
#else
#define VS_GUI_EXPORT __declspec(dllimport)
#endif
#define VS_GUI_TEMPLATE_EXPORT VS_GUI_EXPORT
#else
#error
#endif

#endif  // COMMON_HPP
