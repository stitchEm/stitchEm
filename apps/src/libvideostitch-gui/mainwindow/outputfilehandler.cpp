// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputfilehandler.hpp"

#include <QFileInfo>

ProjectFileHandler::ProjectFileHandler() : workingDirectory(File::getDocumentsLocation()) {}

QString ProjectFileHandler::getFilename() const {
  QMutexLocker locker(&filenameMutex);
  return filename;
}

void ProjectFileHandler::setFilename(QString newFilename) {
  QMutexLocker locker(&filenameMutex);
  workingDirectory = QFileInfo(newFilename).absolutePath();
  filename = newFilename;
}

QString ProjectFileHandler::getWorkingDirectory() const {
  QMutexLocker locker(&filenameMutex);
  return workingDirectory;
}

void ProjectFileHandler::resetFilename() {
  QMutexLocker locker(&filenameMutex);
  filename = QString();
  workingDirectory = File::getDocumentsLocation();
}
