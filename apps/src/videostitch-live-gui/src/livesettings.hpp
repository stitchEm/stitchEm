// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVESETTINGS_HPP
#define LIVESETTINGS_HPP

#include "libvideostitch-gui/mainwindow/vssettings.hpp"

class LiveSettings : public VSSettings {
  Q_OBJECT
 public:
  static LiveSettings* createLiveSettings();  // The application need to be created
  static LiveSettings* getLiveSettings();

  int getLogLevel() const;
  int getOutputConnectionTimeout() const;    // in ms
  int getOutputReconnectionTimeout() const;  // in ms
  bool getShowCalibrationCounter() const;
  bool getMirrorModeEnabled() const;
  QString getSnapshotPath() const;

  void setLogLevel(const int level);
  void setOutputConnectionTimeout(const int timeout);    // in ms
  void setOutputReconnectionTimeout(const int timeout);  // in ms
  void setShowCalibrationCounter(const bool show);
  void setMirrorModeEnabled(const bool mirrorModeEnabled);
  void setSnapshotPath(const QString& path);

 private:
  explicit LiveSettings(const QString& settingsName);
};

#endif  // LIVESETTINGS_HPP
