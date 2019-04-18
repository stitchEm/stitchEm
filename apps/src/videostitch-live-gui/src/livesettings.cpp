// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "livesettings.hpp"
#include "guiconstants.hpp"
#include "libvideostitch-base/common-config.hpp"
#include <QDir>

LiveSettings::LiveSettings(const QString& settingsName) : VSSettings(settingsName) {}

LiveSettings* LiveSettings::createLiveSettings() {
  Q_ASSERT(qApp != nullptr);
  LiveSettings* liveSettings = getLiveSettings();
  if (!liveSettings) {
    liveSettings = new LiveSettings(VAHANA_VR_SETTINGS_NAME);
    liveSettings->setParent(qApp);
  }
  return liveSettings;
}

LiveSettings* LiveSettings::getLiveSettings() {
  Q_ASSERT(qApp != nullptr);
  return qApp->findChild<LiveSettings*>();
}

int LiveSettings::getLogLevel() const { return settings.value("logLevel", 0).toInt(); }

int LiveSettings::getOutputConnectionTimeout() const { return settings.value("output/timeout", 10000).toInt(); }

int LiveSettings::getOutputReconnectionTimeout() const {
  return settings.value("output/reconnectiontimeout", 6500).toInt();
}

bool LiveSettings::getShowCalibrationCounter() const { return settings.value("showCalibrationCounter", true).toBool(); }

bool LiveSettings::getMirrorModeEnabled() const { return settings.value("oculus/mirrorModeEnabled", false).toBool(); }

QString LiveSettings::getSnapshotPath() const {
  QString defaultValue = QDir::toNativeSeparators(getSnapshotsPath());
  return settings.value("snapshotPath", defaultValue).toString();
}

void LiveSettings::setLogLevel(const int level) { settings.setValue("logLevel", level); }

void LiveSettings::setOutputConnectionTimeout(const int timeout) { settings.setValue("output/timeout", timeout); }

void LiveSettings::setOutputReconnectionTimeout(const int timeout) {
  settings.setValue("output/reconnectiontimeout", timeout);
}

void LiveSettings::setShowCalibrationCounter(const bool show) { settings.setValue("showCalibrationCounter", show); }

void LiveSettings::setMirrorModeEnabled(const bool mirrorModeEnabled) {
  settings.setValue("oculus/mirrorModeEnabled", mirrorModeEnabled);
}

void LiveSettings::setSnapshotPath(const QString& path) { settings.setValue("snapshotPath", path); }
