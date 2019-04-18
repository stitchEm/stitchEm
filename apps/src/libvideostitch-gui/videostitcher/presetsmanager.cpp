// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "presetsmanager.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/parse.hpp"

#include <QDir>
#include <QFileSystemWatcher>

PresetsManager::PresetsManager(QObject* parent) : QObject(parent), watcher(nullptr) {
  // Create directories
  QDir dir;
  dir.mkpath(getCustomPresetsFolder());
  dir.mkpath(getRigPresetsFolder());

  loadPresetsFrom(getDefaultPresetsFolder(), false);
  loadPresetsFrom(getCustomPresetsFolder(), true);
  watchCustomPresetsFolder(getCustomPresetsFolder());
}

PresetsManager::~PresetsManager() {}

void PresetsManager::addPreset(const QString& category, const QFileInfo& presetFileInfo, bool custom) {
  QString presetPath = presetFileInfo.filePath();
  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  if (presetPath.isEmpty() || !parser.ok()) {
    return;
  }

  QFile presetFile(presetPath);
  if (!presetFile.open(QFile::ReadOnly)) {
    return;
  }

  QTextStream in(&presetFile);
  std::string content = in.readAll().toStdString();
  if (!parser->parseData(content)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(tr("Could not load preset %0").arg(presetPath));
    return;
  }

  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(tr("Preset file loaded: %0").arg(presetPath));
  if (custom) {
    customPresets[category][presetFileInfo.baseName()].reset(parser->getRoot().clone());
  } else {
    defaultPresets[category][presetFileInfo.baseName()].reset(parser->getRoot().clone());
  }
}

void PresetsManager::watchCustomPresetsFolder(QString dir) {
  Q_ASSERT(watcher == nullptr);
  watcher = new QFileSystemWatcher(this);
  connect(watcher, &QFileSystemWatcher::directoryChanged, this, &PresetsManager::synchronizeCustomPresets);

  QFileInfoList dirs = QDir(dir).entryInfoList(QDir::Dirs);
  foreach (QFileInfo subDir, dirs) { watcher->addPath(subDir.absoluteFilePath()); }
}

void PresetsManager::synchronizeCustomPresets(QString dirPath) {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(tr("Synchronizing preset folder: %0").arg(dirPath));
  QString category = QFileInfo(dirPath).baseName();
  customPresets[category].clear();

  QFileInfoList presetFiles = QDir(dirPath).entryInfoList(QStringList() << "*.preset", QDir::Files);
  foreach (QFileInfo presetFile, presetFiles) { addPreset(category, presetFile, true); }

  emit presetsHasChanged(category);
}

PresetsManager* PresetsManager::getInstance() {
  PresetsManager* presetsManager = qApp->findChild<PresetsManager*>();
  if (presetsManager) {
    return presetsManager;
  } else {
    return new PresetsManager(qApp);
  }
}

QString PresetsManager::rigsCategory() { return QString("rigs"); }

QString PresetsManager::getRigPresetsFolder() {
  return QString("%0/%1").arg(getCustomPresetsFolder()).arg(rigsCategory());
}

bool PresetsManager::hasPreset(QString category, QString name) const {
  return defaultPresets.value(category).contains(name) || customPresets.value(category).contains(name);
}

std::shared_ptr<const VideoStitch::Ptv::Value> PresetsManager::getPresetContent(QString category, QString name) const {
  std::shared_ptr<const VideoStitch::Ptv::Value> preset = defaultPresets.value(category).value(name);
  if (!preset) {
    preset = customPresets.value(category).value(name);
  }
  return preset;
}

std::unique_ptr<VideoStitch::Ptv::Value> PresetsManager::clonePresetContent(QString category, QString name) const {
  std::shared_ptr<const VideoStitch::Ptv::Value> preset = getPresetContent(category, name);
  if (preset) {
    return std::unique_ptr<VideoStitch::Ptv::Value>(preset->clone());
  } else {
    return std::unique_ptr<VideoStitch::Ptv::Value>();
  }
}

QStringList PresetsManager::getPresetNames(QString category, PresetType presetType) const {
  QStringList presetNames;
  if ((presetType & DefaultPresets) == DefaultPresets) {
    presetNames.append(defaultPresets.value(category).keys());
  }
  if ((presetType & CustomPresets) == CustomPresets) {
    presetNames.append(customPresets.value(category).keys());
  }
  return presetNames;
}

QString PresetsManager::getDefaultPresetsFolder() { return QString(":/presets"); }

QString PresetsManager::getCustomPresetsFolder() {
  return QString("%0/%1/Presets").arg(QDir::homePath()).arg(QCoreApplication::applicationName());
}

void PresetsManager::loadPresetsFrom(QString dir, bool custom) {
  QFileInfoList dirs = QDir(dir).entryInfoList(QDir::Dirs);
  foreach (QFileInfo subDir, dirs) {
    QFileInfoList presetFiles = QDir(subDir.filePath()).entryInfoList(QStringList() << "*.preset", QDir::Files);
    foreach (QFileInfo presetFile, presetFiles) { addPreset(subDir.baseName(), presetFile, custom); }
  }
}
