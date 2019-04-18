// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pixelformat.hpp"

void PixelFormatClass::initDescriptions(QMap<Enum, QString>& enumToString) {
  enumToString[VideoStitch::PixelFormat::RGBA] = VideoStitch::getStringFromPixelFormat(VideoStitch::RGBA);
  enumToString[VideoStitch::PixelFormat::RGB] = VideoStitch::getStringFromPixelFormat(VideoStitch::RGB);
  enumToString[VideoStitch::PixelFormat::BGR] = VideoStitch::getStringFromPixelFormat(VideoStitch::BGR);
  enumToString[VideoStitch::PixelFormat::BGRU] = VideoStitch::getStringFromPixelFormat(VideoStitch::BGRU);
  enumToString[VideoStitch::PixelFormat::UYVY] = VideoStitch::getStringFromPixelFormat(VideoStitch::UYVY);
  enumToString[VideoStitch::PixelFormat::YUY2] = VideoStitch::getStringFromPixelFormat(VideoStitch::YUY2);
  enumToString[VideoStitch::PixelFormat::YV12] = VideoStitch::getStringFromPixelFormat(VideoStitch::YV12);
  enumToString[VideoStitch::PixelFormat::Grayscale] = VideoStitch::getStringFromPixelFormat(VideoStitch::Grayscale);
}

const PixelFormatClass::Enum PixelFormatClass::defaultValue = VideoStitch::PixelFormat::RGBA;
