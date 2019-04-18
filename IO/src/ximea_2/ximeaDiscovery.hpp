// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <vector>
#include <xiApi.h>

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/plugin.hpp"

namespace VideoStitch {
namespace Plugin {

class XimeaDiscovery : public VSDiscoveryPlugin {
  struct Device {
    Device() : camIdx(0), autodetection(nullptr) {}

    Plugin::DiscoveryDevice pluginDevice;
    uint8_t camIdx;
    AutoDetection* autodetection;
  };

  struct InputDevice : public Device {
    InputDevice() : Device() {}
  };

 public:
  static XimeaDiscovery* create();
  virtual ~XimeaDiscovery();

  virtual std::string name() const override { return "ximea"; };
  virtual std::string readableName() const override { return "XIMEA"; };

  virtual std::vector<std::string> cards() const override { return {"Ximea System"}; };

  virtual std::vector<Plugin::DiscoveryDevice> inputDevices() override;
  virtual std::vector<Plugin::DiscoveryDevice> outputDevices() override { return {}; };
  virtual void registerAutoDetectionCallback(AutoDetection&) override;
  virtual std::vector<DisplayMode> supportedDisplayModes(const Plugin::DiscoveryDevice&) override;
  virtual std::vector<PixelFormat> supportedPixelFormat(const Plugin::DiscoveryDevice&) override;

  virtual std::vector<int> supportedNbChannels(const Plugin::DiscoveryDevice& device) override { return {}; };
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const Plugin::DiscoveryDevice& device) override {
    return std::vector<Audio::SamplingRate>();
  };
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const Plugin::DiscoveryDevice& device) override {
    return std::vector<Audio::SamplingDepth>();
  };

 private:
  XimeaDiscovery(std::vector<std::shared_ptr<Device>> devices);
  DisplayMode currentDisplayMode(const Plugin::DiscoveryDevice& device);

  uint16_t width;
  uint16_t height;
  float fps;
  std::vector<std::shared_ptr<Device>> m_devices;
};

}  // namespace Plugin
}  // namespace VideoStitch
