// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "formatfactory.hpp"

#include "extensionhandlers/jpgextensionhandler.hpp"
#include "extensionhandlers/pamextensionhandler.hpp"
#include "extensionhandlers/pngextensionhandler.hpp"
#include "extensionhandlers/ppmextensionhandler.hpp"
#include "extensionhandlers/rawextensionhandler.hpp"
#include "extensionhandlers/tiffextensionhandler.hpp"
#include "extensionhandlers/yuv420pextensionhandler.hpp"
#include "movformat.hpp"
#include "mp4format.hpp"
#include "simpleformat.hpp"

Format* FormatFactory::create(const QString& key, QWidget* const parent) {
  if (key == "mp4") {
    return new Mp4Format(parent);
  } else if (key == "mov") {
    return new MovFormat(parent);
  } else if (key == "jpg") {
    return new SimpleFormat(key, new JpgExtensionHandler(), parent);
  } else if (key == "tif") {
    return new SimpleFormat(key, new TiffExtensionHandler(), parent);
  } else if (key == "png") {
    return new SimpleFormat(key, new PngExtensionHandler(), parent);
  } else if (key == "pam") {
    return new SimpleFormat(key, new PamExtensionHandler(), parent);
  } else if (key == "ppm") {
    return new SimpleFormat(key, new PpmExtensionHandler(), parent);
  } else if (key == "raw") {
    return new SimpleFormat(key, new RawExtensionHandler(), parent);
  } else if (key == "yuv420p") {
    return new SimpleFormat(key, new Yuv420pExtensionHandler(), parent);
  } else if (key == "null") {
    return new SimpleFormat(key, nullptr, parent);
  } else {
    return nullptr;
  }
}
