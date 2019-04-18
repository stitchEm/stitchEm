// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MAGEWELL_HDMI_DISCOVERY_HPP_
#define MAGEWELL_HDMI_DISCOVERY_HPP_

#include "libvideostitch/plugin.hpp"

#include <vector>
#include <string>

#include <windows.h>
#include "LibXIStream/XIStream.h"
#include "LibXIProperty/XIProperty.h"

namespace VideoStitch {
namespace Plugin {

class MagewellDiscovery : public VSDiscoveryPlugin {
  struct Device {
    Device()
        : videoCaptureInfo(),
          hVideoCapture(NULL),
          hVideoProperty(NULL),
          autodetection(nullptr),
          supportedDisplayModes() {}

    Plugin::DiscoveryDevice pluginDevice;
    VIDEO_CAPTURE_INFO_EX videoCaptureInfo;
    HANDLE hVideoCapture;
    HANDLE hVideoProperty;
    AutoDetection* autodetection;
    std::vector<DisplayMode> supportedDisplayModes;
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
  static DisplayMode currentDisplayMode(const Device& device);

  std::vector<std::string> m_cards;
  std::vector<Device> m_devices;
};

}  // namespace Plugin
}  // namespace VideoStitch
#endif  // MAGEWELL_HDMI_DISCOVERY_HPP_
