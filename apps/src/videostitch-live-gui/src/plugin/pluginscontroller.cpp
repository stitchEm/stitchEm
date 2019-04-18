// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pluginscontroller.hpp"
#include "guiconstants.hpp"
#include "utils/displaymode.hpp"
#include "utils/pixelformat.hpp"

#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#include "libvideostitch-gui/utils/pluginshelpers.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include <QVector>
#include <set>
#include <algorithm>

void PluginsController::initializePlugins() {
  // Initialize Vahana dedicated plugins
  VideoStitch::Plugin::loadPlugins(
      QString(QApplication::applicationDirPath() + QDir::separator() + LIVE_PLUG_PATH).toStdString());
  // Initialize Common plugins
  VideoStitch::Plugin::loadPlugins(VideoStitch::Plugin::getCorePluginFolderPath().toStdString());
  auto path = VideoStitch::Plugin::getGpuCorePluginFolderPath();
  if (!path.isEmpty()) {
    VideoStitch::Plugin::loadPlugins(path.toStdString());
  }
}

void PluginsController::loadPluginList(QStringList& list) {
  const auto& instances = VideoStitch::Plugin::VSDiscoveryPlugin::Instances();
  for (const VideoStitch::Plugin::VSDiscoveryPlugin* plugin : instances) {
    list << QString::fromStdString(plugin->name());
  }
}

void PluginsController::listDevicesFromPlugin(const QString name, DeviceList& devices) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin) {
    devices = plugin->devices();
  }
}

void PluginsController::listInputDevicesFromPlugin(const QString name, DeviceList& devices) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin) {
    devices = plugin->inputDevices();
  }
}

void PluginsController::listOutputDevicesFromPlugin(const QString name, DeviceList& devices) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin) {
    devices = plugin->outputDevices();
  }
}

std::vector<VideoStitch::Plugin::DisplayMode> PluginsController::listDisplayModes(const QString name,
                                                                                  DeviceList devices) {
  std::vector<VideoStitch::Plugin::DisplayMode> displayModes;
  std::set<VideoStitch::Plugin::DisplayMode> modesSet;
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin != nullptr && !plugin->devices().empty()) {
    if (devices.empty()) {
      devices = plugin->devices();
    }

    // Here we compute the display modes supported by at least one device
    for (const VideoStitch::Plugin::DiscoveryDevice& device : devices) {
      const std::vector<VideoStitch::Plugin::DisplayMode> newDisplayModes = plugin->supportedDisplayModes(device);
      modesSet.insert(newDisplayModes.begin(), newDisplayModes.end());
    }
    displayModes.assign(modesSet.begin(), modesSet.end());
  }
  return displayModes;
}

std::vector<VideoStitch::Plugin::DisplayMode> PluginsController::listDisplayModes(const QString name,
                                                                                  QStringList deviceNames) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  DeviceList devices;
  for (QString deviceName : deviceNames) {
    VideoStitch::Plugin::DiscoveryDevice device = getDeviceByName(plugin, deviceName);
    devices.push_back(device);
  }
  return listDisplayModes(name, devices);
}

VideoStitch::Plugin::DisplayMode PluginsController::currentDisplayMode(const QString name,
                                                                       const unsigned channel) const {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (!plugin) {
    return VideoStitch::Plugin::DisplayMode(0, 0, false, {1, 1});
  }

  const auto devices = plugin->devices();
  if (devices.empty() || devices.size() < channel) {
    return VideoStitch::Plugin::DisplayMode(0, 0, false, {1, 1});
  }
  const auto device = devices[channel];
  return plugin->currentDisplayMode(device);
}

QStringList PluginsController::listNbAudioChannels(const QString name, QString deviceName) {
  QStringList nbAudioChannelsNames;
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin != nullptr && !plugin->devices().empty()) {
    if (deviceName.isEmpty()) {
      deviceName = QString::fromStdString(plugin->devices().front().name);
    }

    VideoStitch::Plugin::DiscoveryDevice device = getDeviceByName(plugin, deviceName);
    std::vector<int> nbChannelsVector = plugin->supportedNbChannels(device);
    std::vector<int> nbAudioChannelsNamesIntersection;
    if (VSSettings::getSettings()->getShowExperimentalFeatures()) {
      // For experimental features list all possible nb input channels
      nbAudioChannelsNamesIntersection = nbChannelsVector;
    } else {
      std::set_intersection(nbChannelsVector.begin(), nbChannelsVector.end(),
                            VideoStitch::AudioHelpers::nbInputChannelsSupported.begin(),
                            VideoStitch::AudioHelpers::nbInputChannelsSupported.end(),
                            std::back_inserter(nbAudioChannelsNamesIntersection));
    }
    for (int nbChannels : nbAudioChannelsNamesIntersection) {
      nbAudioChannelsNames.append(QString::number(nbChannels));
    }
  }
  return nbAudioChannelsNames;
}

QVector<VideoStitch::Audio::SamplingRate> PluginsController::listAudioSamplingRates(const QString name,
                                                                                    QString deviceName) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin != nullptr && !plugin->devices().empty()) {
    if (deviceName.isEmpty()) {
      deviceName = QString::fromStdString(plugin->devices().front().name);
    }

    VideoStitch::Plugin::DiscoveryDevice device = getDeviceByName(plugin, deviceName);
    std::vector<VideoStitch::Audio::SamplingRate> pluginSupportedSamplingRates = plugin->supportedSamplingRates(device);
    std::sort(pluginSupportedSamplingRates.begin(), pluginSupportedSamplingRates.end());
    if (VSSettings::getSettings()->getShowExperimentalFeatures()) {
      return QVector<VideoStitch::Audio::SamplingRate>::fromStdVector(pluginSupportedSamplingRates);
    }
    std::vector<VideoStitch::Audio::SamplingRate> srIntersection;
    std::set_intersection(pluginSupportedSamplingRates.begin(), pluginSupportedSamplingRates.end(),
                          VideoStitch::AudioHelpers::samplingRatesSupported.begin(),
                          VideoStitch::AudioHelpers::samplingRatesSupported.end(), std::back_inserter(srIntersection));
    return QVector<VideoStitch::Audio::SamplingRate>::fromStdVector(srIntersection);
  }
  return QVector<VideoStitch::Audio::SamplingRate>();
}

QList<VideoStitch::Audio::SamplingDepth> PluginsController::listAudioSamplingFormats(const QString name) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin != nullptr && !plugin->devices().empty()) {
    // FIXME: if we have different cards for the same plugin, maybe they will not support the same values
    std::vector<VideoStitch::Audio::SamplingDepth> audioSamplingDepths =
        plugin->supportedSampleFormats(plugin->devices().front());
    return QVector<VideoStitch::Audio::SamplingDepth>::fromStdVector(audioSamplingDepths).toList();
  }
  return QList<VideoStitch::Audio::SamplingDepth>();
}

QStringList PluginsController::listInputDeviceNames(const QString name) {
  QStringList devicesNames;
  DeviceList inputDevices;
  listInputDevicesFromPlugin(name, inputDevices);
  for (const VideoStitch::Plugin::DiscoveryDevice& device : inputDevices) {
    devicesNames << QString::fromStdString(device.displayName);
  }
  return devicesNames;
}

std::vector<std::string> PluginsController::listVideoCodecs(const QString name) {
  VideoStitch::Plugin::VSDiscoveryPlugin* plugin = getPluginByName(name);
  if (plugin != nullptr) {
    return plugin->supportedVideoCodecs();
  }
  return std::vector<std::string>();
}

VideoStitch::Plugin::VSDiscoveryPlugin* PluginsController::getPluginByName(const QString name) {
  std::string pluginName = name.toStdString();
  const auto& instances = VideoStitch::Plugin::VSDiscoveryPlugin::Instances();
  auto it = std::find_if(instances.cbegin(), instances.cend(),
                         [pluginName](const VideoStitch::Plugin::VSDiscoveryPlugin* plugin) -> bool {
                           return plugin->name() == pluginName;
                         });
  if (it != instances.cend()) {
    return *it;
  } else {
    return nullptr;
  }
}

VideoStitch::Plugin::DiscoveryDevice PluginsController::getDeviceByName(VideoStitch::Plugin::VSDiscoveryPlugin* plugin,
                                                                        QString name) {
  if (plugin) {
    std::string stdName = name.toStdString();
    for (const VideoStitch::Plugin::DiscoveryDevice& device : plugin->devices()) {
      if (device.name == stdName) {
        return device;
      }
    }
  }
  return VideoStitch::Plugin::DiscoveryDevice();
}
