// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "extensionhandler.hpp"

class Mp4ExtensionHandler : public ExtensionHandler {
 public:
  Mp4ExtensionHandler() {
    extension = "mp4";
    format = "mp4";
  }

 private:
  QString handle(const QString &filename, const QString &format) const {
    QString ret;
    if (this->format == format) {
      ret = filename;
      ret.remove(FRAME_EXTENSION);
      ret += "." + extension;
    }
    return ret;
  }

  QString stripBasename(const QString &inputText, const QString &format) const {
    QString ret = inputText;
    QString toRemove = QString(".") + extension;
    if (inputText.contains(toRemove) && this->format == format) {
      ret.remove(toRemove);
      return ret;
    } else {
      return QString();
    }
  }
};
