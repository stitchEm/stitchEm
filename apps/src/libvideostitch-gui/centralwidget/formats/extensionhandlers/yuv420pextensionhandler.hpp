// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "extensionhandler.hpp"

class Yuv420pExtensionHandler : public ExtensionHandler {
 public:
  Yuv420pExtensionHandler() {
    extension = "ppm";
    format = "yuv420p";
  }

 private:
  QString handle(const QString &filename, const QString &format) const {
    QString ret;
    if (this->format == format) {
      ret = filename + FRAME_EXTENSION + "y/u/v" + "." + extension;
    }
    return ret;
  }

  QString stripBasename(const QString &inputText, const QString &format) const {
    QString ret = inputText;
    QString toRemove = FRAME_EXTENSION + QString("y/u/v") + QString(".") + extension;
    if (inputText.contains(toRemove) && this->format == format) {
      ret.remove(toRemove);
      return ret;
    } else {
      return QString();
    }
  }
};
