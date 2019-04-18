// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VIDEOSTITCHQOPENGLFUNCTIONS_HPP
#define VIDEOSTITCHQOPENGLFUNCTIONS_HPP

#include <QObject>

#if defined(Q_OS_WIN)
#define VideoStitchQOpenGLFunctions QOpenGLFunctions_3_3_Core
#include <QOpenGLFunctions_3_3_Core>
#else
#define VideoStitchQOpenGLFunctions QOpenGLFunctions
#include <QOpenGLFunctions>
#endif

#endif  // VIDEOSTITCHQOPENGLFUNCTIONS_HPP
