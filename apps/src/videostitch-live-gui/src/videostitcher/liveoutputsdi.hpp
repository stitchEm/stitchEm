// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveoutputfactory.hpp"
#include "utils/displaymode.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

class LiveOutputSDI : public LiveWriterFactory {
  Q_OBJECT
 public:
  explicit LiveOutputSDI(const VideoStitch::Ptv::Value* config, VideoStitch::OutputFormat::OutputFormatEnum type);

  VideoStitch::Potential<VideoStitch::Output::Output> createWriter(LiveProjectDefinition* project,
                                                                   VideoStitch::FrameRate framerate) override;
  virtual QWidget* createStatusWidget(QWidget* const parent) override;

  virtual const QString getIdentifier() const override;
  const VideoStitch::Plugin::DisplayMode& getDisplayMode() const;
  virtual QPixmap getIcon() const override;
  unsigned getHorizontalOffset() const;
  unsigned getVerticalOffset() const;

  void setDeviceName(const QString newDeviceName);
  void setDeviceDisplayName(const QString newDeviceDisplayName);
  void setDisplayMode(VideoStitch::Plugin::DisplayMode newDisplayMode);
  void setHorizontalOffset(const unsigned offset);
  void setVerticalOffset(const unsigned offset);

 private:
  void fillOutputValues(const VideoStitch::Ptv::Value* config);

 protected:
  QString deviceName;
  QString deviceDisplayName;
  VideoStitch::Plugin::DisplayMode displayMode;
  unsigned offset_x;
  unsigned offset_y;
};
