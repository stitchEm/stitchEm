// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/plugin.hpp"

#include "LibMWCapture/MWCapture.h"

#include <vector>
#include <string>
#include <thread>

/**
 * This is the Discovery plugin, based on MWCaptureSDK's
 * LibMWCapture module.
 */

namespace VideoStitch {
namespace Plugin {

class MagewellDiscovery : public VSDiscoveryPlugin {
  struct Device {
    Device()
        : channel(nullptr),
          captureEvent(nullptr),
          videoSignalState(MWCAP_VIDEO_SIGNAL_NONE),
          autodetection(nullptr),
          supportedDisplayModes(),
          listen(true) {}

    Plugin::DiscoveryDevice pluginDevice;
    HCHANNEL channel;
    HANDLE captureEvent;
    MWCAP_VIDEO_SIGNAL_STATE videoSignalState;

    AutoDetection* autodetection;
    std::vector<DisplayMode> supportedDisplayModes;

    bool listen;
  };

 public:
  static MagewellDiscovery* create();
  virtual ~MagewellDiscovery();

  virtual std::string name() const;
  virtual std::string readableName() const;
  virtual std::vector<Plugin::DiscoveryDevice> inputDevices();
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices();
  virtual std::vector<std::string> cards() const;

  virtual void registerAutoDetectionCallback(AutoDetection&);

  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&);
  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice&) {
    return DisplayMode();  // TODO
  }
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&);
  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice& device);
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice& device);
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice& device);

 private:
  MagewellDiscovery();
  void initializeDevices();

  static void signalDetection(Device& device);

  std::vector<std::string> m_cards;
  std::vector<Device> devices;
  std::vector<std::thread*> listeners;
};

}  // namespace Plugin
}  // namespace VideoStitch
