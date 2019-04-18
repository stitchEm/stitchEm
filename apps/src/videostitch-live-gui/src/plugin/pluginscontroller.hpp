// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"

typedef std::vector<VideoStitch::Plugin::DiscoveryDevice> DeviceList;

/**
 * @brief Class for managing the plugin discovery
 */
class PluginsController {
 public:
  /**
   * @brief Loads the list of discovered plugins in list
   * @param list The list of plugins
   */
  static void loadPluginList(QStringList &list);

  /**
   * @brief Loads the core and vahana plugins
   */
  static void initializePlugins();

  /**
   * @brief Asks for a list of devices (connections) for a given plugin
   * @param name A plugin name
   * @param devices A list of devices
   */
  static void listDevicesFromPlugin(const QString name, DeviceList &devices);

  /**
   * @brief Asks for a list of input devices (connections) for a given plugin
   * @param name A plugin name
   * @param devices A list of devices
   */
  static void listInputDevicesFromPlugin(const QString name, DeviceList &devices);

  /**
   * @brief Asks for a list of output devices (connections) for a given plugin
   * @param name A plugin name
   * @param devices A list of devices
   */
  static void listOutputDevicesFromPlugin(const QString name, DeviceList &devices);

  /**
   * @brief Given a plugin, return the current video signal
   * @param name A plugin name
   * @param channel Input video channel
   */
  VideoStitch::Plugin::DisplayMode currentDisplayMode(const QString name, const unsigned channel) const;

  /**
   * @brief Given a plugin and some devices, list and return all the supported display modes
   * If there is no device, all the display modes supported by the plugin are returned.
   * Otherwise, only the display modes supported by at least one device are returned.
   * @param name A plugin name
   * @param devices List of devices
   */
  static std::vector<VideoStitch::Plugin::DisplayMode> listDisplayModes(const QString name,
                                                                        DeviceList devices = DeviceList());

  /**
   * @brief Given a plugin and some devices, list and return all the supported display modes
   * If there is no device, all the display modes supported by the plugin are returned.
   * Otherwise, only the display modes supported by at least one device are returned.
   * @param name A plugin name
   * @param deviceNames List of device names
   */
  static std::vector<VideoStitch::Plugin::DisplayMode> listDisplayModes(const QString name,
                                                                        QStringList deviceNames = QStringList());

  /**
   * @brief Given a plugin, list and return all the supported HW/SW video codecs
   * @param name A plugin name
   */
  static std::vector<std::string> listVideoCodecs(const QString name);

  /**
   * @brief Given a plugin and a device, list and return all the supported nb of channels
   * @param name A plugin name
   * @param deviceName A device name
   */
  static QStringList listNbAudioChannels(const QString name, QString deviceName = QString());

  /**
   * @brief Given a plugin and a device, list and return all the supported sampling rates
   * @param name A plugin name
   * @param deviceName A device name
   */
  static QVector<VideoStitch::Audio::SamplingRate> listAudioSamplingRates(const QString name,
                                                                          QString deviceName = QString());

  /**
   * @brief Given a plugin, list and return all the supported sampling formats
   * @param name A plugin name
   */
  static QList<VideoStitch::Audio::SamplingDepth> listAudioSamplingFormats(const QString name);

  /**
   * @brief Given a plugin, list and return all the input devices readable names
   * @param name A plugin name
   * @return
   */
  static QStringList listInputDeviceNames(const QString name);

  /**
   * @brief Returns a discovered plugin from its id
   * @param name A plugin name
   * @return Nullptr if the plugin was not found, otherwise, the plugin
   */
  static VideoStitch::Plugin::VSDiscoveryPlugin *getPluginByName(const QString name);

  /**
   * @brief Returns a discovered device from its name
   * @param plugin The plugin
   * @param name A device name
   * @return A default built device if the device was not found, otherwise, the device
   */
  static VideoStitch::Plugin::DiscoveryDevice getDeviceByName(VideoStitch::Plugin::VSDiscoveryPlugin *plugin,
                                                              QString name);
};
