// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "extensionhandler.hpp"

class RawExtensionHandler : public ExtensionHandler {
 public:
  RawExtensionHandler() {
    extension = "abgr";
    format = "raw";
  }

 private:
  QString handle(const QString &filename, const QString &format) const {
    QString ret;
    if (this->format == format) {
      ret = filename + "." + extension;
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
