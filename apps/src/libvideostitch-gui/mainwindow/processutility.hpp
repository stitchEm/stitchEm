// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROCESSUTILITY_HPP
#define PROCESSUTILITY_HPP

#include <QString>

#ifdef Q_OS_WIN
#include <Windows.h>
#include <TlHelp32.h>
#elif defined(Q_OS_LINUX)
#include <QFile>
#include <QDir>
#elif defined(Q_OS_MAC)
#include <QFileInfo>
#include <sys/sysctl.h>
#include <libproc.h>
#endif

class VS_GUI_EXPORT ProcessUtility {
 public:
  static int getProcessByName(QString processName);
};

#endif  // PROCESSUTILITY_HPP
