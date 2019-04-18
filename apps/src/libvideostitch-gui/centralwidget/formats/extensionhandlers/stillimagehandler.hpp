// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "extensionhandler.hpp"

class StillImageHandler : public ExtensionHandler {
 protected:
  QString handle(const QString &filename, const QString &format) const {
    QString ret = filename;
    if (this->format == format) {
      if (!filename.contains(FRAME_EXTENSION)) {
        ret += FRAME_EXTENSION;
      }

      return ret + "." + extension;
    } else {
      return QString();
    }
  }

  QString stripBasename(const QString &inputText, const QString &format) const {
    QString ret = inputText;
    QString toRemove = FRAME_EXTENSION + QString(".") + extension;
    if (inputText.contains(toRemove) && this->format == format) {
      ret.remove(toRemove);
      return ret;
    } else {
      return QString();
    }
  }
};
