// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/ptv.hpp"

#include <QFileSystemWatcher>
#include <QObject>

#include <memory>

class QFileInfo;

/**
 * @brief A class used to manage the presets in the output tab. I may also be used as preview presets.
 *        This class loads the presets found in $INSTALL_DIR/presets.
 *        If your preset is called my_cool_preset.ptv your preset name wille be my_cool_preset.
 *
 */
class VS_GUI_EXPORT PresetsManager : public QObject {
  Q_OBJECT

 public:
  static PresetsManager* getInstance();
  static QString rigsCategory();
  static QString getRigPresetsFolder();

  enum PresetType {
    NoPresets = 0x00,
    DefaultPresets = 0x01,
    CustomPresets = 0x02,
    AllPresets = DefaultPresets | CustomPresets
  };

  /**
   * @brief Check if there is a preset with this name in this category
   */
  bool hasPreset(QString category, QString name) const;
  /**
   * @brief Retrieves the preset content and returns it
   * @return The content of the preset. Read-only
   */
  std::shared_ptr<const VideoStitch::Ptv::Value> getPresetContent(QString category, QString name) const;
  /**
   * @brief Clones the preset content and returns it
   * @return The content of the preset. The caller has the ownership.
   */
  std::unique_ptr<VideoStitch::Ptv::Value> clonePresetContent(QString category, QString name) const;
  QStringList getPresetNames(QString category, PresetType presetType = AllPresets) const;

 signals:
  void presetsHasChanged(QString category);

 private:
  static QString getDefaultPresetsFolder();
  static QString getCustomPresetsFolder();

  explicit PresetsManager(QObject* parent = nullptr);
  ~PresetsManager();
  void loadPresetsFrom(QString dir, bool custom);
  void addPreset(const QString& category, const QFileInfo& presetFileInfo, bool custom);
  void watchCustomPresetsFolder(QString dir);

 private slots:
  void synchronizeCustomPresets(QString dirPath);

 private:
  // map<category, map<name, preset>>
  QMap<QString, QMap<QString, std::shared_ptr<const VideoStitch::Ptv::Value>>> defaultPresets;
  QMap<QString, QMap<QString, std::shared_ptr<const VideoStitch::Ptv::Value>>> customPresets;
  QFileSystemWatcher* watcher;
};
