// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputsdi.hpp"
#include "liveaudio.hpp"
#include "liveprojectdefinition.hpp"

#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-gui/videostitcher/stitchercontroller.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/output.hpp"

#include <QLabel>

LiveOutputSDI::LiveOutputSDI(const VideoStitch::Ptv::Value* config, VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveWriterFactory(type) {
  displayMode.framerate = {25, 1};
  offset_x = 0;
  offset_y = 0;
  fillOutputValues(config);
}

const QString LiveOutputSDI::getIdentifier() const { return deviceName; }

const VideoStitch::Plugin::DisplayMode& LiveOutputSDI::getDisplayMode() const { return displayMode; }

void LiveOutputSDI::setDeviceName(const QString newDeviceName) { deviceName = newDeviceName; }

void LiveOutputSDI::setDeviceDisplayName(const QString newDeviceDisplayName) {
  deviceDisplayName = newDeviceDisplayName;
  emit outputDisplayNameChanged(deviceDisplayName);
}

void LiveOutputSDI::setDisplayMode(VideoStitch::Plugin::DisplayMode newDisplayMode) { displayMode = newDisplayMode; }

QPixmap LiveOutputSDI::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/tv.png"); }

unsigned LiveOutputSDI::getHorizontalOffset() const { return offset_x; }

unsigned LiveOutputSDI::getVerticalOffset() const { return offset_y; }

void LiveOutputSDI::setHorizontalOffset(const unsigned offset) { offset_x = offset; }

void LiveOutputSDI::setVerticalOffset(const unsigned offset) { offset_y = offset; }

QWidget* LiveOutputSDI::createStatusWidget(QWidget* const parent) { return createStatusIcon(parent); }

void LiveOutputSDI::fillOutputValues(const VideoStitch::Ptv::Value* config) {
  std::string deviceNameStr;
  if (VideoStitch::Parse::populateString("Ptv", *config, "filename", deviceNameStr, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    deviceName = QString::fromStdString(deviceNameStr);
  }
  std::string deviceDisplayNameStr;
  VideoStitch::Parse::populateString("Ptv", *config, "device_display_name", deviceDisplayNameStr, false);
  setDeviceDisplayName(QString::fromStdString(deviceDisplayNameStr));

  VideoStitch::Parse::populateInt("Ptv", *config, "width", displayMode.width, true);
  VideoStitch::Parse::populateInt("Ptv", *config, "height", displayMode.height, true);
  VideoStitch::Parse::populateBool("Ptv", *config, "interleaved", displayMode.interleaved, true);
  const VideoStitch::Ptv::Value* fpsValue = config->has("frame_rate");
  if (fpsValue) {
    VideoStitch::Parse::populateInt("Ptv", *fpsValue, "num", displayMode.framerate.num, false);
    VideoStitch::Parse::populateInt("Ptv", *fpsValue, "den", displayMode.framerate.den, false);
  }
  VideoStitch::Parse::populateInt("Ptv", *config, "offset_x", offset_x, false);
  VideoStitch::Parse::populateInt("Ptv", *config, "offset_y", offset_y, false);
}

VideoStitch::Potential<VideoStitch::Output::Output> LiveOutputSDI::createWriter(LiveProjectDefinition* project,
                                                                                VideoStitch::FrameRate framerate) {
  std::unique_ptr<VideoStitch::Ptv::Value> parameters(serialize());

  // create the callback
  return VideoStitch::Potential<VideoStitch::Output::Output>(VideoStitch::Output::create(
      *parameters, getIdentifier().toStdString(), project->getPanoConst()->getWidth(),
      project->getPanoConst()->getHeight(), framerate,
      VideoStitch::Audio::getSamplingRateFromInt(audioConfig->getSamplingRate()),
      VideoStitch::Audio::getSamplingDepthFromString(audioConfig->getSamplingFormat().toStdString().c_str()),
      VideoStitch::Audio::getChannelLayoutFromString(audioConfig->getChannelLayout().toStdString().c_str())));
}
