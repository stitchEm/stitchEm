// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "extensionhandler.hpp"
#include "jpgextensionhandler.hpp"
#include "pamextensionhandler.hpp"
#include "ppmextensionhandler.hpp"
#include "yuv420pextensionhandler.hpp"
#include "mp4extensionhandler.hpp"
#include "movextensionhandler.hpp"
#include "rawextensionhandler.hpp"
#include "tiffextensionhandler.hpp"
#include "pngextensionhandler.hpp"

ExtensionHandler::ExtensionHandler() {}

ExtensionHandler::~ExtensionHandler() {
  while (!handlers.isEmpty()) {
    ExtensionHandler *h = handlers.last();
    delete h;
    handlers.pop_back();
  }
}

void ExtensionHandler::init() {
  addHandler(new JpgExtensionHandler);
  addHandler(new Mp4ExtensionHandler);
  addHandler(new MovExtensionHandler);
  addHandler(new PamExtensionHandler);
  addHandler(new PpmExtensionHandler);
  addHandler(new RawExtensionHandler);
  addHandler(new TiffExtensionHandler);
  addHandler(new PngExtensionHandler);
  addHandler(new Yuv420pExtensionHandler);
}

void ExtensionHandler::addHandler(ExtensionHandler *handler) { handlers.push_back(handler); }

QString ExtensionHandler::handle(const QString &filename, const QString &format) const {
  if (filename.isEmpty()) {
    return QString();
  }

  QString ret;
  foreach (ExtensionHandler *h, handlers) {
    ret = h->handle(filename, format);
    if (!ret.isEmpty() && !ret.isNull()) {
      return ret;
    }
  }
  return ret;
}

QString ExtensionHandler::stripBasename(const QString &inputText, const QString &format) const {
  if (inputText.isEmpty()) {
    return QString();
  }

  QString ret;
  foreach (ExtensionHandler *h, handlers) {
    ret = h->stripBasename(inputText, format);
    if (!ret.isEmpty() && !ret.isNull()) {
      return ret;
    }
  }
  if (ret.isEmpty()) {
    return inputText;
  }
  return ret;
}
