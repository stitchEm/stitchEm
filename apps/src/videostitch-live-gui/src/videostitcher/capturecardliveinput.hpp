// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CAPTURECARDLIVEINPUT_H
#define CAPTURECARDLIVEINPUT_H

#include "liveinputfactory.hpp"
#include "utils/pixelformat.hpp"

#include "libvideostitch/plugin.hpp"

class CaptureCardLiveInput : public LiveInputFactory {
 public:
  explicit CaptureCardLiveInput(const QString& name);
  ~CaptureCardLiveInput();

  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);

  VideoStitch::Plugin::DisplayMode getDisplayMode() const;
  PixelFormatEnum getPixelFormat() const;

  void setDisplayMode(const VideoStitch::Plugin::DisplayMode& newDisplayMode);
  void setPixelFormat(const PixelFormatEnum& newPixelFormat);

 protected:
  VideoStitch::Plugin::DisplayMode displayMode;
  PixelFormatEnum pixelFormat;
};

#endif  // CAPTURECARDLIVEINPUT_H
