// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTSTREAM_HPP
#define LIVEINPUTSTREAM_HPP

#include "liveinputfactory.hpp"
#include <QStringList>

class LiveInputStream : public LiveInputFactory {
 public:
  explicit LiveInputStream(const QString& name);
  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);
  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;

  void setWidth(qint64 newWidth) { width = newWidth; }

  void setHeight(qint64 newHeight) { height = newHeight; }

 private:
  // We want to keep the width and the height defined by the stream
  qint64 width;
  qint64 height;
};

#endif  // LIVEINPUTSTREAM_HPP
