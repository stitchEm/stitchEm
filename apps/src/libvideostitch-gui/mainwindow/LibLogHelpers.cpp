// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "LibLogHelpers.hpp"

#include <iostream>
#include <streambuf>

#include <QCoreApplication>

namespace VideoStitch {
namespace Helper {

/**
 * English error messages are provided by VideoStitch::Status
 */
QString createTitle(const VideoStitch::Status& status) {
  return QString::fromStdString(status.getTypeString() + " occurred in " + status.getOriginString());
}

QString createErrorBacktrace(const VideoStitch::Status& status) {
  QString originString = QString::fromStdString(status.getOriginString());
  QString typeString = QString::fromStdString(status.getTypeString());
  QString errorMessage = QString::fromStdString(status.getErrorMessage());

  QString fullMessage = QString("<b>[%0] %1</b><br>%2").arg(originString).arg(typeString).arg(errorMessage);
  if (status.hasCause()) {
    fullMessage += "<br><br>";
    fullMessage += createErrorBacktrace(status.getCause());
  }
  return fullMessage;
}

}  // namespace Helper
}  // namespace VideoStitch
