// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "vssettings.hpp"

#include <QLocale>
#include <QStringList>
#include <QDateTime>
#include <QVector>

static const unsigned int MAX_RECENT_FILES(5);

VSSettings::VSSettings(const QString& settingsName)
    : QObject(),
      settings(
#ifdef Q_OS_MAC
          QCoreApplication::organizationDomain(),
#else
          QCoreApplication::organizationName(),
#endif
          settingsName),
      recentFiles(MAX_RECENT_FILES) {
}

VSSettings* VSSettings::getSettings() {
  Q_ASSERT(qApp != nullptr);
  return qApp->findChild<VSSettings*>();
}

bool VSSettings::contains(const QString& key) const { return settings.contains(key); }

QVariant VSSettings::getValue(const QString& key, const QVariant& defaultValue) const {
  return settings.value(key, defaultValue);
}

void VSSettings::setValue(const QString& key, const QVariant& value) { settings.setValue(key, value); }

bool VSSettings::getIsStereo() const { return settings.value("is-stereo", false).toBool(); }

int VSSettings::getRecentFileNumber() const { return recentFiles; }

QString VSSettings::getLanguage() const {
  QString defaultValue = QLocale::system().name().section('_', 0, 0);
  return settings.value("General/Language", defaultValue).toString();
}

QVector<int> VSSettings::getCudaDevices() const {
  QVariant dev = settings.value("Cuda/Devices");
  QVector<int> devices;
  if (!dev.isValid()) {
    QVariant oldDeviceValue = settings.value("Cuda/Device");
    devices.append(oldDeviceValue.toInt());
  } else {
    for (QVariant v : dev.toList()) {
      bool ok = false;
      int value = v.toInt(&ok);
      if (ok) {
        devices.append(value);
      }
    }
  }
  return devices;
}

QVector<int> VSSettings::getDevices() const {
  QVariant dev = settings.value("gpu-devices");
  QVector<int> devices;

  for (QVariant v : dev.toList()) {
    bool ok = false;
    int value = v.toInt(&ok);
    if (ok) {
      devices.append(value);
    }
  }
  // backward compatibility
  if (devices.empty()) {
    return getCudaDevices();
  }
  return devices;
}

int VSSettings::getMainDevice() const { return getDevices()[0]; }

QDateTime VSSettings::getLastUpdate() const { return settings.value("Update/LastUpdate").toDateTime(); }

QString VSSettings::getLastOpenedFile() const {
  QStringList list = settings.value("recentFileList").toStringList();
  return list.isEmpty() ? QString() : list.first();
}

QStringList VSSettings::getRecentFileList() const { return settings.value("recentFileList").toStringList(); }

QStringList VSSettings::getRecentCalibrationList() const {
  return settings.value("recentCalibrationList").toStringList();
}

bool VSSettings::getIsDumpingCalibrationPictures() const { return settings.value("dumpCalibrationSnapshots").toBool(); }

bool VSSettings::getShowExperimentalFeatures() const { return settings.value("showExperimentalFeatures").toBool(); }

void VSSettings::setRecentFileNumber(int recentFileNumber) { recentFiles = recentFileNumber; }

void VSSettings::setRecentFileList(QStringList list) { settings.setValue("recentFileList", list); }

void VSSettings::setRecentCalibrationList(QStringList list) { settings.setValue("recentCalibrationList", list); }

void VSSettings::setDevices(QVector<int> devices) {
  QVariantList variantDevices;
  for (auto device : devices) {
    variantDevices.append(device);
  }
  settings.setValue("gpu-devices", variantDevices);
}

void VSSettings::setLanguage(QString language) { settings.setValue("General/Language", language); }

void VSSettings::setLastUpdate(QDateTime lastUpdate) { settings.setValue("Update/LastUpdate", lastUpdate); }

void VSSettings::setUuid(QUuid uuid) { settings.setValue("UUID", uuid.toString()); }

void VSSettings::addRecentFile(QString file) { setRecentFileList(addRecent(file, getRecentFileList())); }

void VSSettings::addRecentCalibration(QString file) {
  setRecentCalibrationList(addRecent(file, getRecentCalibrationList()));
}

QStringList VSSettings::addRecent(QString const& file, QStringList files) {
  files.removeAll(file);
  files.prepend(file);
  while (files.size() > recentFiles) {
    files.removeLast();
  }
  return files;
}
