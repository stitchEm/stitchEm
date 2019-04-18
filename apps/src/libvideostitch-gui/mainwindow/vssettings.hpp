// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSSETTINGS_HPP
#define VSSETTINGS_HPP

#include "libvideostitch-gui/common.hpp"

#include <QSettings>
#include <QUuid>

/**
 * @brief The VSSettings class is a base class. It enables your application to be easily configurable.
 *
 * Generic configuration
 * If your configuration is used by only one class, it is recommended to use the 2 generic functions getValue and
 * setValue. This allows to keep the keys and the default values in their context.
 *
 * Specific configuration
 * If your configuration is shared by several classes, you can define specific getters and setters in VSSettings or the
 * derived class.
 *
 * To use VSSettings in a new application, create a derived class with a factory.
 * The VSSettings must be a child of the QCoreApplication and is supposed to be a singleton.
 */
class VS_GUI_EXPORT VSSettings : public QObject {
  Q_OBJECT

 public:
  static VSSettings* getSettings();

  //! @name Getter & setter for generic configuration
  //@{
  bool contains(const QString& key) const;
  QVariant getValue(const QString& key, const QVariant& defaultValue = QVariant()) const;
  void setValue(const QString& key, const QVariant& value);
  //@}

  bool getIsStereo() const;
  int getRecentFileNumber() const;
  QString getLanguage() const;
  QVector<int> getDevices() const;
  int getMainDevice() const;
  QDateTime getLastUpdate() const;
  QStringList getRecentFileList() const;
  QStringList getRecentCalibrationList() const;
  bool getIsDumpingCalibrationPictures() const;
  bool getShowExperimentalFeatures() const;

  void setRecentFileNumber(int recentFileNumber);
  void setDevices(QVector<int> devices);
  void setLanguage(QString language);
  void setLastUpdate(QDateTime lastUpdate);
  void setUuid(QUuid uuid);
  void setIsStereo(const bool stereo);

  void addRecentFile(QString file);
  void addRecentCalibration(QString file);
  QString getLastOpenedFile() const;

 protected:
  explicit VSSettings(const QString& settingsName);
  QStringList addRecent(const QString& file, QStringList files);
  void setRecentFileList(QStringList list);
  void setRecentCalibrationList(QStringList list);

  QSettings settings;
  int recentFiles;

 private:
  QVector<int> getCudaDevices() const;
};

#endif  // VSSETTINGS_HPP
