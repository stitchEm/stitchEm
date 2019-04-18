// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGOUTPUTAUDIO_HPP
#define CONFIGOUTPUTAUDIO_HPP

#include <QWidget>
#include "ui_configoutputaudio.h"
#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

class LiveAudio;
class LiveOutputFactory;

class ConfigOutputAudio : public QWidget, public Ui::ConfigOutputAudioClass {
  Q_OBJECT
 public:
  explicit ConfigOutputAudio(QWidget* const parent = nullptr);
  ~ConfigOutputAudio();

  void setLiveAudio(const LiveOutputFactory* newLiveOutputFactory);
  void setType(const VideoStitch::OutputFormat::OutputFormatEnum type);
  void loadParameters();
  void displayMessage(QString message);

  void saveConfiguration() const;

 signals:
  void notifyConfigChanged();

 private:
  void loadStaticValues();

 private slots:
  void onCodecConfigChanged();

 private:
  const LiveOutputFactory* liveOutputFactory;  // We need the output to show its supported values
  // This should be directly in a dedicated GUI plugin but for the moment, we hardcode these values in the factory
  // so we need to know which output we configure
  LiveAudio* audioConfig;
  VideoStitch::OutputFormat::OutputFormatEnum outputType;
};

#endif  // CONFIGOUTPUTAUDIO_HPP
