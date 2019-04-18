// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "guiconstants.hpp"

QString getUserDataPath() {
  return QDir::toNativeSeparators(QDir::homePath()) + QDir::separator() + QCoreApplication::applicationName();
}

QString getRecordingsPath() { return getUserDataPath() + QDir::separator() + "Recordings"; }

QString getProjectsPath() { return getUserDataPath() + QDir::separator() + "Projects"; }

QString getSnapshotsPath() { return getUserDataPath() + QDir::separator() + "Snapshots"; }

QString getDefaultOutputFileName() { return getRecordingsPath() + QDir::separator() + "output"; }
